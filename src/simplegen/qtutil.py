"""Small Qt helpers shared by the dialogs."""

from __future__ import annotations


def retire_worker(owner, worker) -> None:
    """Finish with a dialog's background QThread without deleting it mid-flow.

    Calling ``deleteLater()`` on a finished worker and then dropping the last Python
    reference crashes PySide when the deferred delete runs in an outer (nested) event
    loop, e.g. saving an edit opened from the Scheduled list (segfault in
    QObject::event). Instead wait for the thread to stop and keep the worker referenced
    by its owner; Qt deletes it together with its parent dialog.
    """
    if worker is None:
        return
    worker.wait()
    owner.__dict__.setdefault("_retired_workers", []).append(worker)
