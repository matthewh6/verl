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
"""Agent loop for the YAM station protocol against a simulator behind ``sim_bridge.py``.

One episode is many model calls, and every call's prompt is REBUILT (system + the last K turns as text +
this turn's three camera frames), so the calls do not share a growing token prefix and cannot be one
``prompt | response`` sequence. ``run`` therefore returns ``list[AgentLoopOutput]``, one output per turn,
each a self-contained single generation. This is the form the V1 trainer supports for it:
``AgentLoopWorkerTQ._agent_loop_postprocess`` stores every output under ``{uid}_{session_id}_{index}``,
copies the final output's reward to the others, and ``compute_advantage_for_multi_trajectories`` computes
the GRPO advantage once per session from the final outputs of a prompt's ``rollout.n`` sessions and
broadcasts it to all of the session's turns.
"""

import asyncio
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.yam_pear import protocol
from verl.experimental.yam_pear.artifacts import EpisodeArtifacts
from verl.experimental.yam_pear.dataloader import AGENT_NAME, OutcomeLog
from verl.experimental.yam_pear.sim_client import SimBridgeClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Simulator processes running at once in this worker process, per event loop.
_sim_slots: dict[int, asyncio.Semaphore] = {}


def _slots(limit: int) -> asyncio.Semaphore:
    key = id(asyncio.get_running_loop())
    if key not in _sim_slots:
        _sim_slots[key] = asyncio.Semaphore(limit)
    return _sim_slots[key]


def _plain(value: Any) -> Any:
    """Config containers -> plain Python."""
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


@register(AGENT_NAME)
class YamPearAgentLoop(AgentLoopBase):
    """Runs one station episode per rollout and returns one training sample per turn.

    Args:
        sims: Simulator name -> ``{command, cwd, env, environment_file, episode_args, timeout}``. ``command``
            is the bridge command line for that simulator's own Python environment; ``environment_file``
            is the ENVIRONMENT text for the system prompt (a name in the recipe's ``prompts`` directory or
            a path); ``episode_args`` are extra driver ``Episode`` arguments (e.g. ``lift`` for ABC).
        history_k: Earlier turns kept in the prompt, as text.
        max_tokens: Completion cap per turn (also capped by ``rollout.response_length``).
        outcome_dir: Directory of the episode outcome log read by the curriculum dataloader. Defaults to
            ``data.custom_dataloader.outcome_dir``; no log is written when neither is set.
        max_concurrent_sims: Simulator processes alive at once per agent loop worker.
        artifacts: ``{dir, frames, val_frames}`` for :class:`EpisodeArtifacts`: per-episode ``episode.json`` /
            ``turns.json`` in the drivers' layout, and camera frames for a sample. Disabled unless ``dir`` is set.
    """

    merges_context_messages = False  # every turn is built with build_initial_tokens; nothing is merged

    def __init__(
        self,
        *args,
        sims: Optional[dict[str, Any]] = None,
        history_k: int = 4,
        max_tokens: int = 6144,
        outcome_dir: Optional[str] = None,
        max_concurrent_sims: int = 4,
        artifacts: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sims: dict[str, Any] = _plain(sims) or {}
        self.history_k = int(history_k)
        self.response_length = int(self.rollout_config.response_length)
        self.max_tokens = min(int(max_tokens), self.response_length)
        self.max_concurrent_sims = int(max_concurrent_sims)
        if outcome_dir is None:
            outcome_dir = (self.data_config.get("custom_dataloader") or {}).get("outcome_dir")
        self.outcomes = OutcomeLog(os.path.expanduser(str(outcome_dir))) if outcome_dir else None
        self.artifacts = EpisodeArtifacts(**(_plain(artifacts) or {}))
        self._think_end_id = self._resolve_think_end_id()

    def _resolve_think_end_id(self) -> Optional[int]:
        token_id = self.tokenizer.convert_tokens_to_ids(protocol.THINK_END)
        if not isinstance(token_id, int) or token_id < 0 or token_id == getattr(self.tokenizer, "unk_token_id", None):
            return None
        return token_id

    def _split_completion(self, token_ids: list[int]) -> tuple[str, str]:
        """(reasoning, answer) of a completion, split at its last ``</think>``."""
        ids, reasoning_ids = list(token_ids), []
        if self._think_end_id is not None and self._think_end_id in ids:
            cut = len(ids) - ids[::-1].index(self._think_end_id)
            reasoning_ids, ids = ids[: cut - 1], ids[cut:]
        reasoning = self.tokenizer.decode(reasoning_ids, skip_special_tokens=True) if reasoning_ids else ""
        text_reasoning, answer = protocol.split_reasoning(self.tokenizer.decode(ids, skip_special_tokens=True))
        return reasoning + text_reasoning, answer

    def _answer_text(self, token_ids: list[int]) -> str:
        """The completion after its reasoning: what the station parses and what the history keeps."""
        return self._split_completion(token_ids)[1]

    def _environment_text(self, sim_config: dict[str, Any]) -> str:
        path = Path(os.path.expanduser(str(sim_config["environment_file"])))
        if not path.is_absolute():
            path = protocol.PROMPT_DIR / path
        return path.read_text()

    @staticmethod
    def _decode_images(images: dict[str, str]) -> dict[str, Image.Image]:
        out = {}
        for camera, data in images.items():
            image = Image.open(io.BytesIO(base64.b64decode(data)))
            image.load()
            out[camera] = image.convert("RGB")
        return out

    async def _generate_turn(
        self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], request_id: str, priority: int
    ) -> Optional[tuple[AgentLoopOutput, TokenOutput]]:
        """One model call on a freshly built prompt. Returns None when the prompt does not fit."""
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        mm_processor_kwargs = self._get_mm_processor_kwargs(None)
        self._assert_mm_supported(bool(multi_modal_data))
        try:
            prompt_ids = await self.ct_build_initial_tokens(messages, images=images)
        except ValueError as exc:
            if "rollout.prompt_length" in str(exc):  # the drivers' context-overflow case
                logger.warning("turn prompt does not fit: %s", exc)
                return None
            raise

        metrics: dict[str, Any] = {}
        start = time.monotonic()
        output: TokenOutput = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params={**sampling_params, "max_tokens": self.max_tokens},
            image_data=images,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
        )
        metrics["generate_sequences"] = time.monotonic() - start
        metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if output.stop_reason == "aborted" or not output.token_ids:
            # says nothing about the policy (the drivers' ``server_error``): fail the rollout
            raise RuntimeError(f"generation returned no usable completion (stop_reason={output.stop_reason!r})")

        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            output.token_ids,
            [],
            [] if output.log_probs else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
        prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        turn = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            # each output is one generation on its own prompt: user + assistant, like the single-turn loop
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields=dict(output.extra_fields),
        )
        return turn, output

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> list[AgentLoopOutput]:
        """Run one episode.

        Args:
            sampling_params: LLM sampling params.
            priority: Request priority assigned by the agent loop manager.
            **kwargs: Dataset fields; ``extra_info`` holds sim, task, seed, rung, prepose and max_turns.

        Returns:
            One AgentLoopOutput per model call, all carrying the episode's best shaped reward.
        """
        priority = int(priority)
        info = dict(kwargs["extra_info"])
        sim, task, seed, rung = str(info["sim"]), str(info["task"]), int(info["seed"]), int(info["rung"])
        prepose, max_turns = str(info["prepose"]), int(info["max_turns"])
        if sim not in self.sims:
            raise KeyError(f"simulator {sim!r} is not configured; configured: {sorted(self.sims)}")
        sim_config = self.sims[sim]
        request_id = uuid4().hex

        outputs: list[AgentLoopOutput] = []
        past: list[tuple[str, str]] = []
        feedback: Optional[str] = None
        end_reason: Optional[str] = None
        completion_tokens = 0
        records: list[dict[str, Any]] = []  # turns.json, in the drivers' field names
        keep_frames = self.artifacts.wants_frames(info)
        frames: dict[str, bytes] = {}
        episode_start = time.monotonic()
        async with _slots(self.max_concurrent_sims):
            client = SimBridgeClient(
                list(sim_config["command"]),
                cwd=sim_config.get("cwd"),
                env=sim_config.get("env"),
                timeout=float(sim_config.get("timeout", 600.0)),
            )
            try:
                reply = await client.reset(sim, task, seed, prepose, sim_config.get("episode_args"))
                instruction, prepose_report = reply["instruction"], reply.get("prepose")
                system = protocol.build_system_prompt(self._environment_text(sim_config), instruction)
                for turn_index in range(max_turns):
                    text = protocol.build_user_text(reply["observation"]["state"], feedback)
                    images = self._decode_images(reply["observation"]["images"])
                    record: dict[str, Any] = {"turn": turn_index, "state": reply["observation"]["state"]}
                    record.update(user_text=text, images=[])
                    records.append(record)
                    if keep_frames:  # exactly the bytes the simulator sent
                        for camera, data in reply["observation"]["images"].items():
                            record["images"].append(f"turn{turn_index:02d}_{camera}.jpg")
                            frames[record["images"][-1]] = base64.b64decode(data)
                    messages = protocol.build_messages(
                        system, past, self.history_k, protocol.build_user_content(text, images)
                    )
                    generated = await self._generate_turn(messages, sampling_params, request_id, priority)
                    if generated is None:
                        record["error"] = "the prompt does not fit the context (rollout.prompt_length)"
                        end_reason = "budget"
                        break
                    turn, token_output = generated
                    outputs.append(turn)
                    completion_tokens += len(token_output.token_ids)
                    reasoning, answer = self._split_completion(token_output.token_ids)
                    record.update(
                        model_seconds=round(turn.metrics.generate_sequences, 2),
                        reasoning=reasoning,
                        answer=answer,
                        finish_reason="length" if token_output.stop_reason == "length" else "stop",
                        usage={"prompt_tokens": len(turn.prompt_ids), "completion_tokens": len(token_output.token_ids)},
                    )
                    if token_output.stop_reason == "length":  # cut at the token limit
                        record["parse_error"] = "the server cut the answer at its token limit"
                        end_reason = "budget"
                        break
                    try:
                        action = protocol.parse_action(answer)
                    except protocol.ActionParseError as exc:
                        record["parse_error"] = str(exc)
                        end_reason = "parse"
                        break
                    start = time.monotonic()
                    reply = await client.step(action)
                    turn.metrics.tool_calls = time.monotonic() - start
                    record.update(
                        action=action,
                        feedback=reply["feedback"],
                        result=reply["result"],
                        sim_seconds=round(turn.metrics.tool_calls, 2),
                        success=bool(reply["success"]),
                        max_reward=float(reply["reward"]),
                    )
                    feedback = reply["feedback"]
                    past.append((text, answer.strip()))
                    if reply["success"]:
                        end_reason = "success"
                        break
                else:
                    end_reason = "max_turns"
                if keep_frames:
                    for camera, data in reply["observation"]["images"].items():
                        frames[f"final_{camera}.jpg"] = base64.b64decode(data)
            finally:
                await client.close()

        reward, success = float(reply["reward"]), bool(reply["success"])
        if self.artifacts.enabled:
            global_steps, session_id = int(kwargs.get("global_steps", 0)), int(kwargs.get("session_id", 0))
            step_dir, name = self.artifacts.location(info, global_steps, session_id)
            episode = {
                "task": task,
                "seed": seed,
                "randomization_request": "",
                "start": "home",
                "instruction": instruction,
                "model": os.path.basename(str(self.config.actor_rollout_ref.model.get("path", "")).rstrip("/")),
                "server": "verl",
                "effort": self.apply_chat_template_kwargs.get("reasoning_effort"),
                "history_k": self.history_k,
                "object_state": False,
                **dict(sim_config.get("episode_args") or {}),  # ABC: lift, lift_pick (the replay tool reads them)
                "prepose": prepose_report,
                "max_tokens": self.max_tokens,
                "end_reason": end_reason,
                "success": success,
                "max_reward": reward,
                "num_turns": len(records),
                "wall_s": round(time.monotonic() - episode_start, 1),
                "completion_tokens": completion_tokens,
                # not in the drivers' episode.json: where the episode sits in the verl run
                "sim": sim,
                "rung": rung,
                "prepose_request": prepose,
                "max_turns": max_turns,
                "split": info.get("split", "train"),
                "global_steps": global_steps,
                "plan_id": int(info.get("plan_id", -1)),
                "calibration": bool(info.get("calibration", False)),
                "uid": str(kwargs.get("uid", "")),
                "session_id": session_id,
                "frames": keep_frames,
            }
            self.artifacts.write(step_dir, name, episode, records, system, frames if keep_frames else None)
        reward_extra_info = {
            "acc": float(success),
            "rung": float(rung),
            "episode_turns": float(len(outputs)),
            "completion_tokens": float(completion_tokens),
            **{f"end_{reason}": float(end_reason == reason) for reason in protocol.END_REASONS},
        }
        for turn in outputs:
            turn.reward_score = reward
            turn.extra_fields["reward_extra_info"] = reward_extra_info
            turn.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        if self.outcomes is not None:
            self.outcomes.append(
                {
                    "split": info.get("split", "train"),
                    "plan_id": int(info.get("plan_id", -1)),
                    "calibration": bool(info.get("calibration", False)),
                    "sim": sim,
                    "task": task,
                    "seed": seed,
                    "rung": rung,
                    "success": success,
                    "reward": reward,
                    "end_reason": end_reason,
                    "turns": len(outputs),
                    "completion_tokens": completion_tokens,
                    "uid": str(kwargs.get("uid", "")),
                    "session_id": int(kwargs.get("session_id", 0)),
                }
            )
        return outputs
