"""Sample N answers per validation question from an HF checkpoint (vLLM) and write per-question correct counts.

Same prompt template, sampling (temperature 1, top-p 1, 400 new tokens) and scorer as the online run's validation.
Output JSON: {"n": N, "rows": [{"gen", "level", "idx", "correct"}]}; pass@k is computed from these in `metrics.py`.
"""
import argparse, json, os, sys
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # the FlashInfer sampler JIT needs nvcc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--val", default=os.path.expanduser("~/datasets/tutor_rg/val.parquet"))
    p.add_argument("--n", type=int, default=16); p.add_argument("--mem", type=float, default=0.4); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default=os.path.expanduser("~/projects/tutor-curriculum"))
    a = p.parse_args()
    if os.path.exists(a.out): return
    os.environ.pop("TC_OUTCOMES", None); sys.path.insert(0, a.project)
    import pandas as pd
    from vllm import LLM, SamplingParams
    from verl_rg.reward import compute_score

    df = pd.read_parquet(a.val)
    kv = {"gpu_memory_utilization": a.mem}
    if os.environ.get("TC_KV_BYTES"): kv["kv_cache_memory_bytes"] = int(os.environ["TC_KV_BYTES"])   # fixed KV budget on a GPU shared with other jobs
    llm = LLM(model=a.ckpt, max_model_len=1424, seed=a.seed, dtype="bfloat16", **kv)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([dict(m) for m in r], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in df["prompt"]]
    outs = llm.generate(prompts, SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=400, seed=a.seed))
    rows = []
    for (_, r), o in zip(df.iterrows(), outs):
        ei = dict(r["extra_info"]); gt = r["reward_model"]["ground_truth"]
        c = sum(int(compute_score(r["data_source"], x.text, gt, extra_info=ei)["acc"] > 0) for x in o.outputs)
        rows.append(dict(gen=ei["gen"], level=int(ei["level"]), idx=int(ei["idx"]), correct=c))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(n=a.n, ckpt=a.ckpt, rows=rows), open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    print("pass@1", sum(r["correct"] for r in rows) / (a.n * len(rows)))


if __name__ == "__main__":
    main()
