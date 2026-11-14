"""Verified append-only session journal and legacy diagnostic snapshots.

The journal assumes ONE externally established writer (desktop bootstrap, then
the known controller CPU). GET/PUT/LIST checks are integrity checks, not an S3
lease, compare-and-swap primitive, or distributed lock.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import threading
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def _session_id(value: str) -> str:
    if not isinstance(value, str) or uuid.UUID(value).hex != value:
        raise ValueError("Use a canonical lowercase UUID hex session ID")
    return value


def _s3_client(region: str):
    if not isinstance(region, str) or not re.fullmatch(
        r"[A-Za-z]+-[A-Za-z0-9]+-\d+", region
    ):
        raise ValueError("Invalid S3 region")
    return boto3.client(
        "s3",
        endpoint_url=f"https://s3api-{region.lower()}.runpod.io",
        region_name=region.lower(),
        aws_access_key_id=os.environ["SG_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["SG_S3_SECRET_KEY"],
        config=Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=10,
            retries={"total_max_attempts": 1, "mode": "standard"},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


class JournalError(RuntimeError):
    """Journal unavailable or invalid; callers must not continue paid mutations."""


class JournalConflict(JournalError):
    """The durable journal differs from this writer's verified history."""


class JournalUncertain(JournalError):
    """A pending revision was not durably verified; reload before proceeding."""


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _credential_free(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Journal JSON object keys must be strings")  # noqa: TRY004 -- invalid serialized state
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if (
                any(word in normalized for word in ("password", "secret", "credential"))
                or normalized.endswith(
                    (
                        "apikey",
                        "accesskey",
                        "accesskeyid",
                        "token",
                        "authorization",
                        "privatekey",
                        "username",
                    )
                )
                or normalized
                in {
                    "authorization",
                    "apikey",
                    "accesskey",
                    "accesskeyid",
                    "sessiontoken",
                    "accesstoken",
                    "refreshtoken",
                    "token",
                    "env",
                    "environment",
                    "controltoken",
                }
            ):
                raise ValueError(
                    "Credentials and deployment environments must not enter journal state"
                )
            _credential_free(child)
    elif isinstance(value, list):
        for child in value:
            _credential_free(child)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("Journal state must contain only JSON values")


class S3Ledger:
    """Single-writer full-state revision journal using only GET, PUT, and LIST.

    Construction/reload validates the entire selected history. Failed integrity or
    I/O checks latch this instance closed. An unresolved PUT is never issued again;
    reload must find its exact revision before this instance may proceed. Recovery
    on another process is permitted only after orchestration establishes that the
    previous writer has stopped. No cached read is returned while recovery is due.
    """

    MAX_BYTES = 2_000_000
    ENVELOPE_SCHEMA = 1

    def __init__(
        self,
        client,
        volume_id: str,
        prefix: str = "simplegen-manager/sessions",
        *,
        session_id: str | None = None,
    ):
        if not isinstance(volume_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,100}", volume_id
        ):
            raise ValueError("Invalid journal volume ID")
        if not isinstance(prefix, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", prefix
        ):
            raise ValueError("Journal prefix must be a relative dedicated S3 prefix")
        self.client, self.volume_id, self.prefix = client, volume_id, prefix
        self.session_id = _session_id(session_id) if session_id is not None else None
        self._states: dict[str, dict] = {}
        self._heads: dict[str, dict] = {}
        self._keys: dict[str, set[str]] = {}
        self._blocked = True
        self._pending: tuple[str, bytes] | None = None
        self._lock = threading.RLock()  # Same-process serialization only.
        self.reload()

    @classmethod
    def from_env(
        cls,
        volume_id: str,
        region: str,
        prefix: str = "simplegen-manager/sessions",
        *,
        session_id: str | None = None,
    ):
        return cls(_s3_client(region), volume_id, prefix, session_id=session_id)

    def _scope(self, sid: str) -> str:
        sid = _session_id(sid)
        if self.session_id is not None and sid != self.session_id:
            raise ValueError("This journal is restricted to a different session")
        return sid

    def _ensure_ready(self) -> None:
        if self._blocked:
            raise JournalUncertain(
                "Journal is blocked; reload and reconcile before further operations"
            )

    def _list(self, prefix: str) -> set[str]:
        keys: set[str] = set()
        token = None
        tokens = set()
        while True:
            args = {"Bucket": self.volume_id, "Prefix": prefix}
            if token is not None:
                args["ContinuationToken"] = token
            try:
                page = self.client.list_objects_v2(**args)
            except Exception:  # noqa: BLE001 -- sanitize all SDK/transport error details
                raise JournalError(
                    "Cannot list the durable journal; recovery is required"
                ) from None
            if not isinstance(page, dict) or type(page.get("IsTruncated")) is not bool:
                raise JournalError("Invalid journal listing pagination")
            contents = page.get("Contents", [])
            if not isinstance(contents, list):
                raise JournalError("Invalid journal listing contents")
            for entry in contents:
                key = entry.get("Key") if isinstance(entry, dict) else None
                if (
                    not isinstance(key, str)
                    or not key.startswith(prefix)
                    or key in keys
                ):
                    raise JournalConflict(
                        "Journal listing contains invalid or duplicate keys"
                    )
                keys.add(key)
            if not page["IsTruncated"]:
                return keys
            token = page.get("NextContinuationToken")
            if not isinstance(token, str) or not token or token in tokens:
                raise JournalError(
                    "Journal pagination is missing a fresh continuation token"
                )
            tokens.add(token)

    def _get(self, key: str, *, missing_ok: bool = False) -> bytes | None:
        try:
            response = self.client.get_object(Bucket=self.volume_id, Key=key)
            stream = response["Body"]
            try:
                body = stream.read(self.MAX_BYTES + 1)
            finally:
                stream.close()
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if missing_ok and code in {"NoSuchKey", "NotFound", "404"}:
                return None
            raise JournalError(
                "Cannot read the durable journal; recovery is required"
            ) from None
        except Exception:  # noqa: BLE001 -- malformed SDK responses must also fail closed
            raise JournalError(
                "Cannot read the durable journal; recovery is required"
            ) from None
        if not isinstance(body, bytes) or not body or len(body) > self.MAX_BYTES:
            raise JournalConflict("Journal revision has an invalid size or body")
        return body

    def _key(self, record: dict) -> str:
        return f"{self.prefix}/{record['session_id']}/revisions/{record['sequence']:020d}-{record['sha256']}.json"

    def _decode(self, key: str, body: bytes) -> dict:
        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("Duplicate key")
                result[name] = value
            return result

        try:
            record = json.loads(body, object_pairs_hook=unique)
            if not isinstance(record, dict) or set(record) != {
                "schema_version",
                "session_id",
                "sequence",
                "previous_sha256",
                "state",
                "sha256",
            }:
                raise ValueError("Invalid envelope")
            if (
                type(record["schema_version"]) is not int
                or record["schema_version"] != self.ENVELOPE_SCHEMA
            ):
                raise ValueError("Unknown envelope schema")
            self._scope(record["session_id"])
            if (
                type(record["sequence"]) is not int
                or not 1 <= record["sequence"] < 10**20
            ):
                raise ValueError("Invalid sequence")
            previous = record["previous_sha256"]
            if previous is not None and (
                not isinstance(previous, str)
                or not re.fullmatch(r"[a-f0-9]{64}", previous)
            ):
                raise ValueError("Invalid previous digest")
            self._validate_state(record["state"])
            if record["state"]["id"] != record["session_id"]:
                raise ValueError("Mismatched session")
            digest = hashlib.sha256(
                _canonical({k: v for k, v in record.items() if k != "sha256"})
            ).hexdigest()
            if (
                record["sha256"] != digest
                or self._key(record) != key
                or body != _canonical(record)
            ):
                raise ValueError("Digest, key or encoding mismatch")
            return record
        except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
            raise JournalConflict(
                "Journal revision is corrupt, incompatible, or contains credentials"
            ) from None

    def _validate_state(self, state: dict) -> None:
        if (
            not isinstance(state, dict)
            or type(state.get("schema")) is not int
            or state["schema"] != 2
        ):
            raise ValueError(
                "Journal requires controller state schema 2; legacy snapshots cannot be adopted"
            )
        self._scope(state.get("id"))
        _credential_free(state)
        _canonical(state)

    def reload(self) -> None:
        """Verify all revisions, then atomically replace cached state and unblock."""
        with self._lock:
            self._blocked = True
            prefix = f"{self.prefix}/" + (
                f"{self.session_id}/" if self.session_id else ""
            )
            keys = self._list(prefix)
            if any(not known.issubset(keys) for known in self._keys.values()):
                raise JournalConflict(
                    "Previously verified journal revisions disappeared; refusing rollback"
                )
            pattern = re.compile(
                re.escape(self.prefix)
                + r"/([a-f0-9]{32})/revisions/([0-9]{20})-([a-f0-9]{64})\.json"
            )
            grouped: dict[str, list[tuple[int, str]]] = {}
            for key in keys:
                match = pattern.fullmatch(key)
                if not match:
                    raise JournalConflict(
                        "Legacy snapshots or unknown objects exist under the journal prefix"
                    )
                sid = self._scope(match[1])
                grouped.setdefault(sid, []).append((int(match[2]), key))
            states, heads, verified_keys = {}, {}, {}
            for sid, revisions in grouped.items():
                previous = None
                verified_keys[sid] = set()
                for expected, (sequence, key) in enumerate(sorted(revisions), 1):
                    if sequence != expected:
                        raise JournalConflict(
                            "Journal revision sequence contains a gap or conflicting branch"
                        )
                    body = self._get(key)
                    record = self._decode(key, body)
                    if record["previous_sha256"] != previous:
                        raise JournalConflict("Journal previous-digest chain is broken")
                    previous = record["sha256"]
                    states[sid], heads[sid] = record["state"], record
                    verified_keys[sid].add(key)
                    if (
                        self._pending is not None
                        and key == self._pending[0]
                        and body != self._pending[1]
                    ):
                        raise JournalConflict(
                            "Pending journal write differs from its durable revision"
                        )
            if self._pending is not None and self._pending[0] not in keys:
                raise JournalUncertain(
                    "Pending journal write is still unresolved; do not write or provision"
                )
            self._states, self._heads, self._keys = states, heads, verified_keys
            self._pending = None
            self._blocked = False

    def get(self, sid: str) -> dict:
        with self._lock:
            self._ensure_ready()
            return copy.deepcopy(self._states[self._scope(sid)])

    def all(self) -> list[dict]:
        with self._lock:
            self._ensure_ready()
            return [copy.deepcopy(self._states[sid]) for sid in sorted(self._states)]

    def save(self, state: dict) -> None:
        with self._lock:
            self._ensure_ready()
            self._validate_state(state)
            state = copy.deepcopy(state)
            sid = state["id"]
            previous = self._heads.get(sid)
            record = {
                "schema_version": self.ENVELOPE_SCHEMA,
                "session_id": sid,
                "sequence": previous["sequence"] + 1 if previous else 1,
                "previous_sha256": previous["sha256"] if previous else None,
                "state": state,
            }
            record["sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
            body = _canonical(record)
            if len(body) > self.MAX_BYTES:
                raise ValueError("Manager journal revision unexpectedly large")
            key = self._key(record)
            try:
                remote_keys = self._list(f"{self.prefix}/{sid}/")
                if remote_keys != self._keys.get(sid, set()):
                    raise JournalConflict(
                        "Journal changed outside this writer; reload and verify CPU ownership"
                    )
                if previous is not None:
                    prior_key = self._key(previous)
                    if self._get(prior_key) != _canonical(previous):
                        raise JournalConflict(
                            "Previous journal revision changed or disappeared"
                        )
                if self._get(key, missing_ok=True) is not None:
                    raise JournalConflict(
                        "Unexpected existing journal revision; refusing overwrite"
                    )
                self._pending = (key, body)
                try:
                    self.client.put_object(
                        Bucket=self.volume_id,
                        Key=key,
                        Body=body,
                        ContentType="application/json",
                        ContentMD5=base64.b64encode(
                            hashlib.md5(body).digest()
                        ).decode(),
                    )
                except Exception:  # noqa: BLE001, S110 -- verify exact readback instead of exposing SDK details
                    # A lost PUT response is resolved only by reading this exact
                    # object. Never repeat PUT or advance to a different revision.
                    pass
                readback = self._get(key, missing_ok=True)
                if readback is None:
                    raise JournalUncertain(
                        "Journal write outcome is unresolved; reload before proceeding"
                    )
                if readback != body:
                    raise JournalConflict(
                        "Journal write readback differs from the intended revision"
                    )
                self._decode(key, readback)
                expected_keys = remote_keys | {key}
                if self._list(f"{self.prefix}/{sid}/") != expected_keys:
                    raise JournalConflict(
                        "Journal changed during write verification; writer ownership must be checked"
                    )
            except Exception:
                self._blocked = True
                raise
            self._states[sid], self._heads[sid] = state, record
            self._keys[sid] = expected_keys
            self._pending = None


class StateStore:
    def __init__(self, client, volume_id: str):
        self.client, self.volume_id = client, volume_id

    @classmethod
    def from_env(cls, volume_id: str, region: str):
        return cls(_s3_client(region), volume_id)

    def snapshot(self, sid: str, state: dict):
        # Controller state is credential-free; worker env must never be passed here.
        _credential_free(state)
        key = f"simplegen-manager/sessions/{_session_id(sid)}/status.json"
        body = json.dumps(state, allow_nan=False, sort_keys=True).encode()
        if len(body) > 2_000_000:
            raise ValueError("Manager snapshot unexpectedly large")
        self.client.put_object(
            Bucket=self.volume_id,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentMD5=base64.b64encode(hashlib.md5(body).digest()).decode(),
        )
