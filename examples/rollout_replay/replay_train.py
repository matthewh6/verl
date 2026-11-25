"""Train a student from the base model on an archive of online-RL rollouts (arXiv 2611.33102).

The archive is the per-step files written by `trainer.rollout_archive_dir` (one file = the exact batch the
online run trained on at that step).  The student takes ONE optimizer update per archived batch.

  --loss sft        token-level NLL on every response, no rewards              (paper Eq. 11)
  --loss sft_pos    same, but only responses with reward > 0 (filtered BC; not in the paper)
  --loss pg         the online policy-gradient update replayed offline          (paper Eq. 8)
  --loss pg_is      pg with per-token truncated importance ratios, C = --is-clip (paper Eq. 9-10)

  --order chrono | reverse | shuffle | last:<m>   (last:<m> cycles over the final m batches)

Checkpoints (HF format, bf16) are written at --save-at update counts for `eval_ckpt.py`.
"""
import argparse, glob, json, os, random, shutil, time
import torch, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("--archive", required=True); p.add_argument("--out", required=True)
p.add_argument("--model", default=os.path.expanduser("~/datasets/models/Qwen3.5-0.8B"))
p.add_argument("--loss", default="sft", choices=["sft", "sft_pos", "pg", "pg_is"])
p.add_argument("--order", default="chrono"); p.add_argument("--updates", type=int, default=0, help="0 = one pass over the chosen batches")
p.add_argument("--first", type=int, default=1); p.add_argument("--last", type=int, default=0, help="use archive steps first..last (0 = all)")
p.add_argument("--lr", type=float, default=2e-6); p.add_argument("--wd", type=float, default=0.01); p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--is-clip", type=float, default=2.0); p.add_argument("--micro", type=int, default=4)
p.add_argument("--save-at", default="", help="comma list of update counts to checkpoint at (the last update is always saved)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed)

files = sorted(glob.glob(os.path.join(a.archive, "step_*.pt")))
steps = [int(os.path.basename(f)[5:11]) for f in files]
files = [f for f, s in zip(files, steps) if s >= a.first and (a.last == 0 or s <= a.last)]
assert files, "empty archive"
if a.order == "chrono": plan = list(files)
elif a.order == "reverse": plan = list(reversed(files))
elif a.order == "shuffle": plan = list(files); random.shuffle(plan)
elif a.order.startswith("last:"): plan = files[-int(a.order.split(":")[1]):]
else: raise SystemExit(f"unknown order {a.order}")
n_updates = a.updates or len(plan)
plan = [plan[i % len(plan)] for i in range(n_updates)]
save_at = {int(x) for x in a.save_at.split(",") if x} | {n_updates}

from transformers import AutoConfig
import transformers
cfg = AutoConfig.from_pretrained(a.model)
cls = getattr(transformers, cfg.architectures[0])
model = cls.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
model.gradient_checkpointing_enable(); model.train()
body, head = model.model, model.lm_head
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.999))


def rows(d, key):
    v = d.get(key)
    if v is None: return None
    return list(v) if isinstance(v, list) else [r for r in v]


def load(path):
    d = torch.load(path, weights_only=False)
    keep = [i for i, pad in enumerate(d["is_padding"]) if not pad]
    out = {}
    for k in ("prompts", "responses", "rm_scores", "old_log_probs", "rollout_log_probs", "advantages"):
        r = rows(d, k); out[k] = [r[i] for i in keep] if r is not None else None
    out["reward"] = [float(x.sum()) for x in out["rm_scores"]]
    return out


def save(tag):
    dst = os.path.join(a.out, f"ckpt_{tag:04d}"); os.makedirs(dst, exist_ok=True)
    sd = model.state_dict()
    for f in os.listdir(a.model):
        src = os.path.join(a.model, f)
        if f.endswith(".safetensors"):
            with safe_open(src, "pt") as s:
                t = {k: (sd[k].detach().to("cpu", torch.bfloat16) if k in sd else s.get_tensor(k)) for k in s.keys()}
                missing = [k for k in s.keys() if k not in sd]
            save_file(t, os.path.join(dst, f), metadata={"format": "pt"})
            if missing: print(f"note: {len(missing)} tensors copied from base (not in the trained module), e.g. {missing[0]}")
        elif os.path.isfile(src): shutil.copy(src, dst)
    return dst


log = open(os.path.join(a.out, "train.jsonl"), "a"); t0 = time.time()
json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
for u, path in enumerate(plan, 1):
    d = load(path); n = len(d["responses"])
    use = [i for i in range(n) if len(d["responses"][i]) > 0 and (a.loss != "sft_pos" or d["reward"][i] > 0)]
    n_tok = sum(len(d["responses"][i]) for i in use)
    opt.zero_grad(set_to_none=True); tot = 0.0; nll = 0.0; rho_sum = 0.0; rho_clipped = 0.0
    use.sort(key=lambda i: len(d["prompts"][i]) + len(d["responses"][i]))
    for b in range(0, len(use), a.micro):
        idx = use[b:b + a.micro]
        seqs = [torch.cat([d["prompts"][i], d["responses"][i]]).long() for i in idx]
        L = max(len(s) for s in seqs)
        ids = torch.zeros(len(idx), L, dtype=torch.long); att = torch.zeros(len(idx), L, dtype=torch.long); tgt = torch.zeros(len(idx), L, dtype=torch.bool)
        for j, (i, s) in enumerate(zip(idx, seqs)):
            ids[j, :len(s)] = s; att[j, :len(s)] = 1; tgt[j, len(d["prompts"][i]):len(s)] = True
        ids, att, tgt = ids.cuda(), att.cuda(), tgt.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = body(input_ids=ids, attention_mask=att).last_hidden_state
            sel = tgt[:, 1:]                       # position t predicts token t+1
            logits = head(h[:, :-1][sel])
        logp = -F.cross_entropy(logits.float(), ids[:, 1:][sel], reduction="none")   # response tokens, row-major
        if a.loss in ("sft", "sft_pos"):
            w = torch.ones_like(logp)
        else:
            w = torch.cat([d["advantages"][i].float() for i in idx]).cuda()
            if a.loss == "pg_is":
                old = torch.cat([d["old_log_probs"][i].float() for i in idx]).cuda()
                ratio = torch.exp(logp.detach() - old)
                rho = ratio.clamp(max=a.is_clip); w = w * rho
                rho_sum += float(rho.sum()); rho_clipped += float((ratio > a.is_clip).sum())
        loss = -(w * logp).sum() / max(n_tok, 1)
        loss.backward(); tot += float(loss); nll += float(-logp.detach().sum())
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)); opt.step()
    rec = dict(update=u, src=os.path.basename(path), n=len(use), tokens=n_tok, loss=tot, nll=nll / max(n_tok, 1), grad_norm=gn,
               mean_reward=sum(d["reward"]) / n, t=round(time.time() - t0, 1))
    if a.loss == "pg_is": rec.update(rho_mean=rho_sum / max(n_tok, 1), rho_clipped=rho_clipped / max(n_tok, 1))
    log.write(json.dumps(rec) + "\n"); log.flush()
    if u % 10 == 0 or u == 1: print(rec, flush=True)
    if u in save_at: print("saved", save(u), flush=True)
