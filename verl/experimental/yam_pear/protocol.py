# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The YAM station conversation, as the PEAR drivers hold it (``raiden_pear.py`` / ``abc_pear_cur.py`` ``main()``).

Pure Python (no torch, no verl): the system prompt, the per-turn user text, the SHORT-CONTEXT message list
(system + the last K turns as text + this turn's camera images; rebuilt every turn, not a growing prefix),
and the parsing of the model's answer into an action.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_TEMPLATE_FILE = "station_system_prompt.md"

JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
THINK_END = "</think>"
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."
ARM_Q = ("x", "y", "z", "roll", "pitch", "yaw")
SCHEMA = {"left_arm": ARM_Q, "right_arm": ARM_Q, "left_gripper": ("width",), "right_gripper": ("width",)}

# How an episode ends. ``success``/``max_turns``/``parse``/``budget`` are the drivers' end reasons; the
# drivers' ``server_error`` has no value here because the agent loop raises instead (verl then marks the
# rollout as failed, which is the counterpart of the orchestrator redoing such episodes).
END_REASONS = ("success", "max_turns", "parse", "budget")


class ActionParseError(ValueError):
    """The answer does not end with a valid fenced json action (end reason ``parse``)."""


def build_system_prompt(environment: str, instruction: str, template: Optional[str] = None) -> str:
    """Fill the station's system prompt template.

    Args:
        environment: The simulator's ENVIRONMENT text (``*_environment.md``).
        instruction: The task instruction the simulator reports at reset.
        template: The template text, default the recipe's ``station_system_prompt.md``.

    Returns:
        The system prompt.
    """
    if template is None:
        template = (PROMPT_DIR / SYSTEM_TEMPLATE_FILE).read_text()
    environment_block = f"## ENVIRONMENT\n\n{environment.strip()}\n\n"
    return template.replace("{environment}", environment_block).replace("{instruction}", instruction)


def build_user_text(state: dict[str, Any], feedback: Optional[str]) -> str:
    """The text of a turn's user message: the measured state, the reminder, and the last execution feedback."""
    text = json.dumps(state) + "\n\n" + REMINDER
    if feedback:
        text += "\n\nPrevious execution: " + feedback
    return text


def build_user_content(text: str, images: dict[str, Any]) -> list[dict[str, Any]]:
    """The current turn's user content: its text, then ``[camera]`` and the frame for every camera, in order."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for camera, image in images.items():
        content += [{"type": "text", "text": f"[{camera}]"}, {"type": "image", "image": image}]
    return content


def build_messages(
    system: str, past: list[tuple[str, str]], history_k: int, user_content: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The short-context prompt of one turn.

    Args:
        system: The system prompt.
        past: (user text, assistant answer) of every earlier turn whose action was carried out.
        history_k: How many of the latest earlier turns the prompt keeps, as text (no images, no reasoning).
        user_content: This turn's user content from :func:`build_user_content`.

    Returns:
        The message list: system, the kept history, this turn's user message.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for user_text, answer in past[max(0, len(past) - history_k) :] if history_k > 0 else []:
        messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": user_content})
    return messages


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a completion into (reasoning, answer) at the last ``</think>``; no marker means all answer."""
    if THINK_END in text:
        reasoning, _, answer = text.rpartition(THINK_END)
        return reasoning, answer
    return "", text


def check_action(action: Any) -> None:
    """Raise ValueError unless the action gives exactly the station's parts and finite quantities."""
    if not isinstance(action, dict) or set(action) != set(SCHEMA):
        raise ValueError("the json block must map exactly the parts " + ", ".join(SCHEMA))
    for part, names in SCHEMA.items():
        q = action[part]
        if not isinstance(q, dict) or set(q) != set(names):
            raise ValueError(f"{part!r} must give exactly its quantities {', '.join(names)}")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in q.values()):
            raise ValueError("action quantities must be finite numbers")


def parse_action(answer: str) -> dict[str, Any]:
    """Parse the final fenced json action of an answer.

    Args:
        answer: The completion after its reasoning (see :func:`split_reasoning`).

    Returns:
        The action.

    Raises:
        ActionParseError: No json block, text after the last fence, invalid json, or a wrong schema.
    """
    blocks = JSON_BLOCK.findall(answer)
    if not blocks or answer[answer.rfind("```") + 3 :].strip():
        raise ActionParseError("Final answer must end with a fenced json action")
    try:
        action = json.loads(blocks[-1])
        check_action(action)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ActionParseError(str(exc)) from None
    return action
