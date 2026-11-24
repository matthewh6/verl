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
"""Station protocol: system prompt, short-context prompt building, action parsing."""

import json

import pytest

from verl.experimental.yam_pear import protocol

ACTION = {
    "left_arm": {"x": 0, "y": 0, "z": -5.0, "roll": 0, "pitch": 0, "yaw": 0},
    "right_arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
    "left_gripper": {"width": -1},
    "right_gripper": {"width": 0},
}


def fenced(action) -> str:
    return "```json\n" + json.dumps(action) + "\n```"


@pytest.mark.parametrize("environment_file", ["raiden_environment.md", "abc_environment.md", "fake_environment.md"])
def test_system_prompt_fills_the_station_template(environment_file):
    environment = (protocol.PROMPT_DIR / environment_file).read_text()
    system = protocol.build_system_prompt(environment, "pick up the cube")
    template = (protocol.PROMPT_DIR / protocol.SYSTEM_TEMPLATE_FILE).read_text()
    assert "{environment}" in template and "{instruction}" in template
    assert "{environment}" not in system and "{instruction}" not in system
    assert f"## ENVIRONMENT\n\n{environment.strip()}\n\n" in system
    assert "pick up the cube" in system


def test_user_text_matches_the_driver():
    state = {"left_arm": {"x": 1.0}}
    assert protocol.build_user_text(state, None) == json.dumps(state) + "\n\n" + protocol.REMINDER
    assert protocol.build_user_text(state, "") == json.dumps(state) + "\n\n" + protocol.REMINDER
    with_feedback = protocol.build_user_text(state, '{"status": "completed"} Use the new measured observation.')
    assert with_feedback.endswith(
        protocol.REMINDER + '\n\nPrevious execution: {"status": "completed"} Use the new measured observation.'
    )


def test_user_content_is_text_then_labelled_frames_in_camera_order():
    content = protocol.build_user_content("T", {"head": "H", "left_wrist": "L", "right_wrist": "R"})
    assert content == [
        {"type": "text", "text": "T"},
        {"type": "text", "text": "[head]"},
        {"type": "image", "image": "H"},
        {"type": "text", "text": "[left_wrist]"},
        {"type": "image", "image": "L"},
        {"type": "text", "text": "[right_wrist]"},
        {"type": "image", "image": "R"},
    ]


@pytest.mark.parametrize("n_past", [0, 1, 4, 5, 9])
@pytest.mark.parametrize("history_k", [0, 1, 4])
def test_short_context_keeps_only_the_last_k_turns_as_text(n_past, history_k):
    past = [(f"user {i}", f"answer {i}") for i in range(n_past)]
    content = protocol.build_user_content("now", {"head": object()})
    messages = protocol.build_messages("SYS", past, history_k, content)

    # the driver's own slice (raiden_pear.py main): past[len(past) - K:] if K else []
    reference = past[len(past) - history_k :] if history_k else []
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[-1] == {"role": "user", "content": content}
    middle = messages[1:-1]
    assert len(middle) == 2 * min(n_past, history_k) == 2 * len(reference)
    for (user, answer), u_msg, a_msg in zip(reference, middle[0::2], middle[1::2], strict=True):
        assert u_msg == {"role": "user", "content": user}  # plain text: no images in the history
        assert a_msg == {"role": "assistant", "content": answer}
    images = [
        part for m in messages if isinstance(m["content"], list) for part in m["content"] if part["type"] == "image"
    ]
    assert len(images) == 1  # only this turn's frames


def test_prompt_is_rebuilt_not_a_growing_prefix():
    past = [(f"user {i}", f"answer {i}") for i in range(6)]
    turn5 = protocol.build_messages("SYS", past[:5], 4, [{"type": "text", "text": "t5"}])
    turn6 = protocol.build_messages("SYS", past[:6], 4, [{"type": "text", "text": "t6"}])
    assert len(turn5) == len(turn6) == 1 + 8 + 1
    assert turn6[: len(turn5) - 1] != turn5[:-1]  # turn 1 left the window
    assert turn5[1]["content"] == "user 1" and turn6[1]["content"] == "user 2"


def test_split_reasoning():
    assert protocol.split_reasoning("plan a</think>plan b</think>\nanswer") == ("plan a</think>plan b", "\nanswer")
    assert protocol.split_reasoning("only answer") == ("", "only answer")


def test_parse_action_accepts_a_final_fenced_block():
    assert protocol.parse_action("I will descend.\n" + fenced(ACTION) + "\n  ") == ACTION
    two = fenced({**ACTION, "left_gripper": {"width": 1}}) + "\nactually:\n" + fenced(ACTION)
    assert protocol.parse_action(two) == ACTION  # the last block is the action


@pytest.mark.parametrize(
    "answer",
    [
        "no block at all",
        fenced(ACTION) + "\nand then some trailing words",
        "```json\n{not json}\n```",
        "```\n" + json.dumps(ACTION) + "\n```",  # not tagged json
        fenced({k: v for k, v in ACTION.items() if k != "right_gripper"}),
        fenced({**ACTION, "extra": {}}),
        fenced({**ACTION, "left_arm": {"x": 0}}),
        fenced({**ACTION, "left_gripper": {"width": "closed"}}),
        fenced({**ACTION, "left_gripper": {"width": True}}),
        "```json\n" + json.dumps(ACTION).replace("-5.0", "NaN") + "\n```",
        "```json\n[1, 2]\n```",
    ],
)
def test_parse_action_rejects(answer):
    with pytest.raises(protocol.ActionParseError):
        protocol.parse_action(answer)


def test_end_reasons_are_the_drivers():
    assert protocol.END_REASONS == ("success", "max_turns", "parse", "budget")
