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
"""Opt-in check against a recorded episode of the pipeline this recipe replaces (skipped by default).

Needs files that are not in the repository:

    YAM_PEAR_MODEL_DIR    a directory with the PEAR model's tokenizer, processor and chat template
                          (no weights are loaded)
    YAM_PEAR_EPISODE_DIR  an episode directory written by raiden_pear.py with --token-ids --history-k 4
                          (turns.json, episode.json, system_prompt.md, turnNN_<camera>.jpg)

It rebuilds every turn's prompt with this recipe's protocol code and verl's Continuous Token builder and
compares the token ids with the ``prompt_token_ids`` vLLM's chat-completions endpoint reported for that turn.
"""

import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.experimental.yam_pear import protocol

MODEL_DIR = os.environ.get("YAM_PEAR_MODEL_DIR")
EPISODE_DIR = os.environ.get("YAM_PEAR_EPISODE_DIR")
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR and EPISODE_DIR), reason="set YAM_PEAR_MODEL_DIR and YAM_PEAR_EPISODE_DIR"
)


@pytest.mark.asyncio
async def test_prompts_and_answers_match_the_recorded_vllm_episode():
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.yam_pear.agent_loop import YamPearAgentLoop
    from verl.utils.dataset.rl_dataset import RLHFDataset

    episode_dir = Path(EPISODE_DIR)
    episode = json.loads((episode_dir / "episode.json").read_text())
    turns = json.loads((episode_dir / "turns.json").read_text())
    system = (episode_dir / "system_prompt.md").read_text()
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"prompt_length": 32768, "response_length": 6144}},
            "data": {
                "apply_chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": False,
                    "reasoning_effort": episode["effort"],
                }
            },
        }
    )
    loop = YamPearAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=None,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR),
        processor=AutoProcessor.from_pretrained(MODEL_DIR),
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        hf_model_type=AutoConfig.from_pretrained(MODEL_DIR).model_type,
        sims={},
        history_k=episode["history_k"],
    )
    environment = (protocol.PROMPT_DIR / "raiden_environment.md").read_text()
    assert protocol.build_system_prompt(environment, episode["instruction"]) == system

    checked = 0
    for turn in turns:
        if not turn.get("prompt_token_ids"):
            continue
        images = {}
        for name in turn["images"]:
            image = Image.open(episode_dir / name)
            image.load()
            images[name.split("_", 1)[1].rsplit(".", 1)[0]] = image.convert("RGB")
        marker = "\n\nPrevious execution: "
        feedback = turn["user_text"].split(marker, 1)[1] if marker in turn["user_text"] else None
        text = protocol.build_user_text(turn["state"], feedback)
        assert text == turn["user_text"]
        past = [(h["user"], h["assistant"]) for h in turn["history"]]
        messages = protocol.build_messages(system, past, loop.history_k, protocol.build_user_content(text, images))
        multi_modal_data = await loop.process_multi_modal_info(messages)
        prompt_ids = await loop.ct_build_initial_tokens(messages, images=multi_modal_data["images"])
        assert prompt_ids == turn["prompt_token_ids"], f"turn {turn['turn']}: prompt tokens differ from vLLM's"
        if turn.get("token_ids") and turn.get("answer") is not None:
            assert loop._answer_text(turn["token_ids"]).strip() == turn["answer"].strip()
        checked += 1
    assert checked > 0
