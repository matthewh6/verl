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
"""The agent loop end to end: a mocked LLM server, the fake simulator in a subprocess, and verl's own
per-output storage and multi-trajectory GRPO advantage on what the loop returns."""

import asyncio
import json
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, _agent_loop_registry
from verl.experimental.yam_pear import protocol
from verl.experimental.yam_pear.agent_loop import YamPearAgentLoop
from verl.experimental.yam_pear.artifacts import EpisodeArtifacts
from verl.experimental.yam_pear.dataloader import AGENT_NAME, OutcomeLog, make_row
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput

IMAGE_TOKEN = 0x1F5BC  # one placeholder "token" per camera frame
THINK_END_ID = 0x1F4AD
TOPK = 4


def action(dz: float) -> dict:
    zero = {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}
    return {
        "left_arm": {**zero, "z": dz},
        "right_arm": zero,
        "left_gripper": {"width": 0},
        "right_gripper": {"width": 0},
    }


def reply(dz: float, reasoning: str = "lower the arm") -> str:
    return reasoning + protocol.THINK_END + "\n\nNext action:\n```json\n" + json.dumps(action(dz)) + "\n```"


class CharTokenizer:
    """One token per character; ``</think>`` is a single token, as in the real vocabulary."""

    unk_token_id = None

    def encode(self, text: str) -> list[int]:
        ids = []
        for i, part in enumerate(text.split(protocol.THINK_END)):
            ids += ([THINK_END_ID] if i else []) + [ord(c) for c in part]
        return ids

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join(protocol.THINK_END if i == THINK_END_ID else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return THINK_END_ID if token == protocol.THINK_END else None


class FakeBuilder:
    """Stands in for the Continuous Token builder: renders messages to character tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def supports_multimodal(self):
        return True

    def build_initial_tokens(self, messages, tools=None, images=None, videos=None, audios=None, mm_inputs_out=None):
        ids, n_images = [], 0
        for message in messages:
            ids += self.tokenizer.encode(f"<{message['role']}>")
            parts = (
                message["content"]
                if isinstance(message["content"], list)
                else [{"type": "text", "text": message["content"]}]
            )
            for part in parts:
                if part["type"] == "image":
                    ids.append(IMAGE_TOKEN)
                    n_images += 1
                else:
                    ids += self.tokenizer.encode(part["text"])
        assert n_images == len(images or [])
        return ids + self.tokenizer.encode("<assistant>")

    def merge_assistant_tokens(self, runtime_token_ids, assistant_token_ids):
        return OmegaConf.create(
            {"token_ids": list(runtime_token_ids) + list(assistant_token_ids), "n": len(assistant_token_ids)}
        )

    def align_response_metadata(self, merge_result, response_mask, response_logprobs, assistant_logprobs=None):
        mask = list(response_mask) + [1] * merge_result.n
        logprobs = None if assistant_logprobs is None else list(response_logprobs or []) + list(assistant_logprobs)
        return mask, logprobs


class MockServer:
    """LLMServerClient stand-in: answers from a policy(prompt text, call index) and records every request."""

    def __init__(self, tokenizer, policy, stop_reason="completed"):
        self.tokenizer, self.policy, self.stop_reason = tokenizer, policy, stop_reason
        self.calls = []

    async def generate(
        self, request_id, *, prompt_ids, sampling_params, image_data=None, mm_processor_kwargs=None, priority=0
    ):
        prompt = self.tokenizer.decode(prompt_ids)
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), "prompt": prompt,
                           "sampling_params": dict(sampling_params), "images": image_data})  # fmt: skip
        stop = self.stop_reason(len(self.calls) - 1) if callable(self.stop_reason) else self.stop_reason
        ids = self.tokenizer.encode(self.policy(prompt, len(self.calls) - 1))
        if stop == "aborted":
            ids = []
        n = len(ids)
        topk_ids = (
            np.stack([np.array(ids)] + [np.arange(1, n + 1) + j for j in range(1, TOPK)], axis=1)
            if n
            else np.zeros((0, TOPK))
        )
        return TokenOutput(
            token_ids=ids,
            log_probs=[-0.25] * n,
            stop_reason=stop,
            num_preempted=0,
            extra_fields={
                "response_topk_ids": topk_ids,
                "response_topk_log_probs": np.tile(np.log(np.array([0.7788, 0.1, 0.05, 0.02])), (n, 1)),
                "response_support_complete": np.zeros(n, dtype=np.bool_),
                "min_global_steps": 3,
                "max_global_steps": 3,
            },
        )


def make_loop(tmp_path, server, history_k=2, max_tokens=4096, artifacts=None):
    loop = YamPearAgentLoop.__new__(YamPearAgentLoop)
    tokenizer = server.tokenizer
    loop.tokenizer = tokenizer
    loop.processor = object()  # multimodal on: frames are extracted from the messages
    loop.dataset_cls = RLHFDataset
    loop.data_config = OmegaConf.create({})
    loop.config = OmegaConf.create({"actor_rollout_ref": {"model": {"path": "/models/Qwen3.8-27B-PEAR/"}}})
    loop.apply_chat_template_kwargs = {"reasoning_effort": "low"}
    loop.artifacts = EpisodeArtifacts(**(artifacts or {}))
    loop.mm_processor_kwargs = {}
    loop.rollout_config = OmegaConf.create({"prompt_length": 100000, "response_length": 2048})
    loop.continuous_token_builder = FakeBuilder(tokenizer)
    loop.server_manager = server
    loop.loop = asyncio.get_running_loop()
    loop.sims = {
        "fake": {
            "command": [sys.executable, "{bridge}", "--backend", "fake"],
            "environment_file": "fake_environment.md",
            "timeout": 30,
        }
    }
    loop.history_k = history_k
    loop.response_length = 2048
    loop.max_tokens = min(max_tokens, 2048)
    loop.max_concurrent_sims = 2
    loop.outcomes = OutcomeLog(tmp_path / "outcomes")
    loop._think_end_id = loop._resolve_think_end_id()
    return loop


def row_kwargs(rung, seed=5, uid="uid", session_id=0):
    row = make_row("fake", "pick_cube", seed, rung, split="train", index=0, plan_id=9)
    return {
        "raw_prompt": row["prompt"],
        "extra_info": row["extra_info"],
        "uid": uid,
        "session_id": session_id,
        "global_steps": 1,
    }


SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "logprobs": True,
    "topk_log_probs": TOPK,
}


def test_agent_loop_is_registered():
    assert _agent_loop_registry[AGENT_NAME]["_target_"] == "verl.experimental.yam_pear.agent_loop.YamPearAgentLoop"
    assert YamPearAgentLoop.merges_context_messages is False


@pytest.mark.asyncio
async def test_full_episode_short_context_protocol(tmp_path):
    server = MockServer(CharTokenizer(), lambda prompt, i: reply(-5.0, reasoning=f"secret reasoning {i}"))
    loop = make_loop(tmp_path, server, history_k=2)
    outputs = await loop.run(dict(SAMPLING), priority=3, **row_kwargs(rung=11))  # from Home

    n = len(outputs)
    assert 4 <= n <= 8 and len(server.calls) == n  # the fake scene is 4-8 descents from Home
    assert len({call["request_id"] for call in server.calls}) == 1  # one sticky session per episode
    system = protocol.build_system_prompt(
        (protocol.PROMPT_DIR / "fake_environment.md").read_text(), "pick up the cube (pick_cube)"
    )
    answer = "\n\nNext action:\n```json\n" + json.dumps(action(-5.0)) + "\n```"
    for t, call in enumerate(server.calls):
        prompt = call["prompt"]
        assert prompt.startswith("<system>" + system + "<user>")
        assert prompt.count(protocol.REMINDER) == min(t, 2) + 1  # the last K=2 turns as text + this turn
        assert prompt.count("<assistant>") == min(t, 2) + 1
        assert prompt.count(answer.strip()) == min(t, 2)  # the history keeps the answer ...
        assert "secret reasoning" not in prompt  # ... without the reasoning
        assert call["prompt_ids"].count(IMAGE_TOKEN) == 3 and len(call["images"]) == 3  # this turn's frames only
        for camera in ("head", "left_wrist", "right_wrist"):
            assert f"[{camera}]" in prompt
        assert all(image.mode == "RGB" for image in call["images"])
        assert call["sampling_params"]["max_tokens"] == 2048 and call["sampling_params"]["top_p"] == 1.0
        assert "Previous execution" not in prompt  # every action was a plain success: no feedback
    # rebuilt each turn, not a growing prefix
    assert server.calls[3]["prompt_ids"][: len(server.calls[2]["prompt_ids"])] != server.calls[2]["prompt_ids"]
    states = [json.loads(call["prompt"].split("<user>")[-1].split("\n\n")[0])["left_arm"]["z"] for call in server.calls]
    assert states == sorted(states, reverse=True) and len(set(states)) == n  # each turn sees the new observation

    for output, call in zip(outputs, server.calls, strict=True):
        assert output.prompt_ids == call["prompt_ids"]
        assert output.response_ids == server.tokenizer.encode(server.policy("", server.calls.index(call)))
        assert output.response_mask == [1] * len(output.response_ids)
        assert output.response_logprobs == [-0.25] * len(output.response_ids)
        assert output.num_turns == 2 and len(output.multi_modal_data["images"]) == 3
        assert output.reward_score == 1.0  # the episode's reward on every turn
        info = output.extra_fields["reward_extra_info"]
        assert info["acc"] == 1.0 and info["end_success"] == 1.0 and info["episode_turns"] == n and info["rung"] == 11.0
        assert info["end_max_turns"] == info["end_parse"] == info["end_budget"] == 0.0
        assert output.metrics.generate_sequences > 0 and output.metrics.tool_calls > 0
    (outcome,) = OutcomeLog(tmp_path / "outcomes").read_new()
    assert outcome["success"] is True and outcome["reward"] == 1.0 and outcome["end_reason"] == "success"
    assert (outcome["plan_id"], outcome["rung"], outcome["seed"], outcome["turns"]) == (9, 11, 5, n)
    assert outcome["split"] == "train" and outcome["calibration"] is False


@pytest.mark.asyncio
async def test_feedback_reaches_the_next_prompt_and_max_turns_ends_the_episode(tmp_path):
    server = MockServer(CharTokenizer(), lambda prompt, i: reply(0.0))  # never descends
    loop = make_loop(tmp_path, server)
    outputs = await loop.run(dict(SAMPLING), **row_kwargs(rung=0))
    assert len(outputs) == 4  # rung 0's turn cap
    assert "Previous execution" not in server.calls[0]["prompt"]
    for call in server.calls[1:]:
        assert 'Previous execution: {"status": "completed"' in call["prompt"]
        assert call["prompt"].count("Use the new measured observation.") >= 1
    info = outputs[-1].extra_fields["reward_extra_info"]
    assert info["end_max_turns"] == 1.0 and info["acc"] == 0.0
    assert all(o.reward_score == 0.0 for o in outputs)


@pytest.mark.asyncio
async def test_reward_is_the_best_shaped_reward(tmp_path):
    def policy(prompt, i):
        return reply(-5.0) if i == 0 else "I am done."

    server = MockServer(CharTokenizer(), policy)
    loop = make_loop(tmp_path, server)
    outputs = await loop.run(dict(SAMPLING), **row_kwargs(rung=1))  # two descents left; one is made
    assert len(outputs) == 2
    assert [o.reward_score for o in outputs] == [0.5, 0.5]
    info = outputs[-1].extra_fields["reward_extra_info"]
    assert info["end_parse"] == 1.0 and info["acc"] == 0.0
    assert outputs[0].extra_fields["reward_extra_info"] is info or outputs[0].extra_fields["reward_extra_info"] == info


@pytest.mark.parametrize(
    "text",
    ["no action here", reply(-5.0) + "\ntrailing words", "x" + protocol.THINK_END + '```json\n{"left_arm": 1}\n```'],
)
@pytest.mark.asyncio
async def test_parse_end_reason(tmp_path, text):
    server = MockServer(CharTokenizer(), lambda prompt, i: text)
    loop = make_loop(tmp_path, server)
    outputs = await loop.run(dict(SAMPLING), **row_kwargs(rung=3))
    assert len(outputs) == 1 and len(server.calls) == 1  # the unparsed turn is still a training sample
    assert outputs[0].extra_fields["reward_extra_info"]["end_parse"] == 1.0
    assert outputs[0].reward_score == 0.0
    assert OutcomeLog(tmp_path / "outcomes").read_new()[0]["end_reason"] == "parse"


@pytest.mark.asyncio
async def test_action_inside_the_reasoning_is_not_an_action(tmp_path):
    text = "```json\n" + json.dumps(action(-5.0)) + "\n```" + protocol.THINK_END + "I should think more."
    server = MockServer(CharTokenizer(), lambda prompt, i: text)
    loop = make_loop(tmp_path, server)
    outputs = await loop.run(dict(SAMPLING), **row_kwargs(rung=0))
    assert outputs[0].extra_fields["reward_extra_info"]["end_parse"] == 1.0


@pytest.mark.asyncio
async def test_budget_end_reason_when_cut_at_the_token_limit(tmp_path):
    server = MockServer(
        CharTokenizer(), lambda prompt, i: reply(-5.0), stop_reason=lambda i: "length" if i == 1 else "completed"
    )
    loop = make_loop(tmp_path, server)
    outputs = await loop.run(dict(SAMPLING), **row_kwargs(rung=11))
    assert len(outputs) == 2  # a complete-looking answer is still not executed when the server cut it
    info = outputs[-1].extra_fields["reward_extra_info"]
    assert info["end_budget"] == 1.0 and 0.0 < outputs[-1].reward_score < 1.0


@pytest.mark.asyncio
async def test_budget_end_reason_when_the_prompt_does_not_fit(tmp_path):
    server = MockServer(CharTokenizer(), lambda prompt, i: reply(-5.0))
    loop = make_loop(tmp_path, server)
    loop.rollout_config = OmegaConf.create({"prompt_length": 50, "response_length": 2048})
    outputs = await loop.run(dict(SAMPLING), **row_kwargs(rung=0))
    assert outputs == [] and server.calls == []
    assert OutcomeLog(tmp_path / "outcomes").read_new()[0]["end_reason"] == "budget"


@pytest.mark.asyncio
async def test_server_failure_fails_the_rollout_and_stops_the_simulator(tmp_path):
    server = MockServer(CharTokenizer(), lambda prompt, i: reply(-5.0), stop_reason="aborted")
    loop = make_loop(tmp_path, server)
    with pytest.raises(RuntimeError, match="no usable completion"):
        await loop.run(dict(SAMPLING), **row_kwargs(rung=0))
    assert not (tmp_path / "outcomes").exists()  # says nothing about the policy: not an outcome
    with pytest.raises(KeyError, match="not configured"):
        await loop.run(
            dict(SAMPLING), **{**row_kwargs(rung=0), "extra_info": {**row_kwargs(rung=0)["extra_info"], "sim": "nope"}}
        )


@pytest.mark.asyncio
async def test_station_frames_pass_verls_image_preprocessing_unchanged():
    """verl extracts frames with qwen_vl_utils (smart_resize). 640x480 station frames are already a
    multiple of the 32 px unit of a patch-16 processor, so the model gets the pixels the simulator sent."""
    from PIL import Image

    frame = Image.effect_noise((640, 480), 64).convert("RGB")
    messages = [{"role": "user", "content": protocol.build_user_content("t", {"head": frame})}]
    images, _, _ = await RLHFDataset.process_multi_modal_info(
        messages, image_patch_size=16, config=OmegaConf.create({})
    )
    assert images[0].size == (640, 480) and images[0].tobytes() == frame.tobytes()


class _DummyWorker:
    """As tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py: the real postprocess without Ray."""

    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    distillation_enabled = False

    def __init__(self):
        self.rollout_config = OmegaConf.create({"topk_log_probs": TOPK})
        self.processor = None
        self.reward_loop_worker_handles = None


@pytest.mark.asyncio
async def test_group_through_verls_storage_and_grpo_advantage(tmp_path, monkeypatch):
    """Two same-seed sessions with different outcomes: every turn is stored as its own sample with its
    sampler top-k head, and verl's multi-trajectory GRPO gives all turns of a session the session's advantage."""
    stored = {"keys": [], "fields": [], "tags": []}

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        assert partition_id == "train"
        stored["keys"] += keys
        stored["fields"].append(fields)
        stored["tags"] += tags

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", async_kv_batch_put)
    postprocess = AgentLoopWorkerTQ.__ray_metadata__.modified_class._agent_loop_postprocess
    policies = [lambda prompt, i: reply(-5.0), lambda prompt, i: reply(-5.0) if i < 2 else reply(0.0)]
    n_turns = []
    for session_id, policy in enumerate(policies):
        loop = make_loop(tmp_path, MockServer(CharTokenizer(), policy))
        kwargs = row_kwargs(rung=3, uid="group", session_id=session_id)  # three descents left, cap 12
        outputs = await loop.run(dict(SAMPLING), **kwargs)
        n_turns.append(len(outputs))
        await postprocess(_DummyWorker(), outputs, False, **kwargs)
    assert n_turns == [3, 12]
    assert stored["keys"] == [f"group_0_{i}" for i in range(3)] + [f"group_1_{i}" for i in range(12)]
    assert all(tag["status"] == "success" and tag["min_global_steps"] == 3 for tag in stored["tags"])

    rewards, masks, lengths = [], [], []
    for fields in stored["fields"]:
        for i in range(len(fields)):
            prompt_len, response_len = fields["prompts"][i].numel(), fields["responses"][i].numel()
            # score centering: one sampler head per response token, laid over the whole sequence
            assert fields["rollout_topk_ids"][i].shape == (prompt_len + response_len, TOPK)
            assert fields["rollout_topk_ids"][i][prompt_len - 1].tolist()[0] == fields["responses"][i][0].item()
            assert fields["rollout_log_probs"][i].numel() == response_len
            assert fields["rm_scores"][i].sum().item() == pytest.approx(fields["rm_scores"][i][-1].item())
            rewards.append(fields["rm_scores"][i].sum().item())
            masks.append(fields["response_mask"][i])
            lengths.append(response_len)
            assert "response_topk_ids" not in fields["extra_fields"][i]
    assert rewards == pytest.approx([1.0] * 3 + [2 / 3] * 12)

    width = max(lengths)
    token_level_rewards = torch.zeros(len(rewards), width)
    response_mask = torch.zeros(len(rewards), width, dtype=torch.long)
    for i, (reward, mask) in enumerate(zip(rewards, masks, strict=True)):
        token_level_rewards[i, len(mask) - 1] = reward
        response_mask[i, : len(mask)] = mask
    data = DataProto.from_dict(
        tensors={"token_level_rewards": token_level_rewards, "response_mask": response_mask},
        non_tensors={"uid": np.array(["group"] * len(rewards), dtype=object)},
    )
    result = compute_advantage_for_multi_trajectories(
        data, batch_keys=stored["keys"], adv_estimator=AdvantageEstimator.GRPO
    )
    advantages = result.batch["advantages"]
    per_row = [advantages[i, 0].item() for i in range(len(rewards))]
    assert torch.equal(advantages != 0, response_mask.bool())  # on every generated token of every turn
    assert len(set(per_row[:3])) == 1 and len(set(per_row[3:])) == 1
    assert per_row[0] > 0 > per_row[3]
    assert per_row[0] == pytest.approx(-per_row[3], rel=1e-4)  # normalised within the group of two episodes


@pytest.mark.parametrize("name", ["agent_loop.yaml", "agent_loop_fake.yaml"])
def test_shipped_agent_loop_configs_match_the_constructor(name, monkeypatch):
    import inspect
    from pathlib import Path

    import hydra

    from verl.experimental.yam_pear.agent_loop import _plain
    from verl.experimental.yam_pear.sim_client import BRIDGE_SCRIPT, SimBridgeClient

    monkeypatch.setenv("PEAR_ROOT", "/pear")
    monkeypatch.setenv("ABC_SIM_DIR", "/abc")
    path = Path(protocol.PROMPT_DIR).parent / "config" / name
    (config,) = OmegaConf.load(path)
    assert config.name == AGENT_NAME and hydra.utils.get_class(config._target_) is YamPearAgentLoop
    accepted = set(inspect.signature(YamPearAgentLoop.__init__).parameters)
    assert set(config) - {"name", "_target_"} <= accepted
    sims = _plain(config.sims)
    for sim in sims.values():
        assert (protocol.PROMPT_DIR / sim["environment_file"]).is_file()
        client = SimBridgeClient(sim["command"], cwd=sim.get("cwd"), env=sim.get("env"))
        assert str(BRIDGE_SCRIPT) in client.command and "__EGL_VENDOR_LIBRARY_DIRS" not in client.env
    if name == "agent_loop.yaml":
        assert sims["raiden"]["command"][0] == "/pear/raiden/venv/bin/python" and sims["abc"]["cwd"] == "/abc"
        assert sims["abc"]["episode_args"] == {"lift": 0.08, "lift_pick": 0}
