"""H3 Scene Chain dialogue extraction."""
import re
from . import plan


DIALOGUE = re.compile(r"<d>\s*\[[^\]\n]+\]\s*(.*?)</d>", re.S)


GROK_DENIED = ("run_terminal_command,read_file,search_replace,list_dir,grep,write,"
               "x_user_search,x_semantic_search,x_keyword_search,x_thread_fetch,"
               "kill_command_or_subagent,todo_write,get_command_or_subagent_output,"
               "spawn_subagent,scheduler_create,scheduler_delete,scheduler_list,monitor,"
               "search_tool,use_tool,workflow,enter_plan_mode,exit_plan_mode,ask_user_question,"
               "send_feedback,image_gen,image_edit,image_to_video,reference_to_video")


QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class RoomError(RuntimeError):
    pass


def normalize_line(text: str) -> str:
    """Ignore whitespace and typographic/outer quote differences, not punctuation."""
    text = " ".join(text.translate(QUOTES).split())
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def plan_lines(candidate: dict) -> list[str]:
    """Read all six blocks in render order, including accidental extra dialogue."""
    lines = []
    for segment in candidate["segments"]:
        for block in plan.BLOCKS:
            text = segment["sections"].get(block, "")
            if not isinstance(text, str):
                raise RoomError(f"{block} must be text")
            found = list(DIALOGUE.finditer(text))
            if text.count("<d>") != len(found) or text.count("</d>") != len(found):
                raise RoomError("malformed dialogue tag: expected <d>[Language] words</d>")
            lines.extend(normalize_line(match[1]) for match in found)
    return lines
