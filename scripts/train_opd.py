"""On-policy distillation: the student samples, the teacher scores every sampled token,
advantage = teacher_logp - student_logp, one policy-gradient update per rollout batch.

Colab:  %run scripts/train_opd.py --name opd_smoke --steps 20
This script does not touch git; call save(...) from the notebook afterwards.
"""
import argparse, json, os, subprocess, sys, time
try:
    import datasets
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

from src.models import make_mesh, hbm, load, STUDENT_ID, TEACHER_ID   # imports src.env_setup first
from src.lora import apply_lora
from src import eval_gsm8k as ev
from src import opd

import numpy as np, optax, jax, jax.numpy as jnp
from flax import nnx
from datasets import load_dataset
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib, tokenizer_adapter
from tunix.rl import common


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name; logs go to results/runs/<name>/")
    p.add_argument("--steps", type=int, default=230)
    p.add_argument("--rollout-bs", type=int, default=16, help="answers per step; eval uses the same batch size")
    p.add_argument("--mb", type=int, default=4, help="micro-batch for teacher scoring and the gradient pass")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--max-prompt", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--eval-n", type=int, default=192)
    p.add_argument("--ckpt-dir", default=None, help="LoRA checkpoints; default Drive/opd-runs/<name>")
    return p.parse_args(argv)


def main(a):
    run = f"results/runs/{a.name}"; os.makedirs(run, exist_ok=False)   # a new name per run, so logs never mix
    json.dump(vars(a), open(f"{run}/args.json", "w"), indent=2)
    from google.colab import drive; drive.mount("/content/drive")
    ckpt = a.ckpt_dir or f"/content/drive/MyDrive/opd-runs/{a.name}"; os.makedirs(ckpt, exist_ok=True)
    hf_tok = AutoTokenizer.from_pretrained(STUDENT_ID); tok = tokenizer_adapter.TokenizerAdapter(hf_tok)
    PAD = hf_tok.convert_tokens_to_ids("<|endoftext|>"); IM_END = hf_tok.convert_tokens_to_ids("<|im_end|>")
    eos = [IM_END, PAD]
    BS, MB = a.rollout_bs, a.mb
    assert BS % MB == 0 and a.eval_n % BS == 0

    mesh = make_mesh()
    student = apply_lora(load(STUDENT_ID, mesh), mesh, seed=a.seed)
    teacher = load(TEACHER_ID, mesh); hbm("student + LoRA + teacher")

    graphdef, params, rest = nnx.split(student, nnx.LoRAParam, ...)
    n_lora = sum(int(np.prod(np.shape(x))) for x in jax.tree.leaves(params))
    assert n_lora > 0, "no nnx.LoRAParam found: qwix LoRA weights are a different type than expected"
    print(f"trainable LoRA params: {n_lora / 1e6:.2f}M")
    params0 = [np.array(x) for x in jax.tree.leaves(params)]
    print("LoRA param dtypes:", sorted({str(x.dtype) for x in params0}))

    c = student.config
    smp = sampler_lib.Sampler(transformer=student, tokenizer=tok, cache_config=sampler_lib.CacheConfig(
        cache_size=a.max_prompt + a.max_new, num_layers=c.num_layers, num_kv_heads=c.num_kv_heads, head_dim=c.head_dim))
    lora_vars = [v for v in smp.model_def_and_state()[1] if isinstance(v, nnx.LoRAParam)]
    assert lora_vars, "the sampler holds no LoRA variables"
    before = [np.array(opd.val(v)) for v in lora_vars]
    opd.check_random_sampling(smp, ev.make_prompt(hf_tok, "What is 25 * 13? Show your work."), eos, a.max_prompt)
    print("sampler holds", len(lora_vars), "LoRA variables")

    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, a.warmup, a.steps, a.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched))
    opt_state = tx.init(params)
    grad_fn = opd.make_grad_fn(graphdef, PAD, IM_END)
    t_gd, t_st = nnx.split(teacher)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    order = np.random.RandomState(a.seed).permutation(len(ds))
    exam = ev.load_gsm8k_test(a.eval_n)
    cum = {"gen": 0, "scored": 0, "sec": 0.0}

    def do_eval(step):
        t0 = time.time()
        res = ev.evaluate(smp, hf_tok, eos, exam, bs=BS, max_new=a.max_new, max_prompt=a.max_prompt)
        rec = {"step": step, "cum_gen_tokens": cum["gen"], "cum_teacher_scored_tokens": cum["scored"],
               "train_seconds": round(cum["sec"], 1), "eval_seconds": round(time.time() - t0), **res}
        open(f"{run}/eval.jsonl", "a").write(json.dumps(rec) + "\n"); print("EVAL", rec)
        opd.save_lora(params, f"{ckpt}/lora_step{step}.npz")
        return rec

    sync_checked = False
    for step in range(1, a.steps + 1):
        t0 = time.time()
        rows = [ds[int(j)] for j in order[(step - 1) * BS: step * BS]]
        prompts = [ev.make_prompt(hf_tok, r["question"]) for r in rows]
        out = smp(input_strings=prompts, max_generation_steps=a.max_new, max_prompt_length=a.max_prompt,
                  temperature=1.0, top_k=None, top_p=1.0, seed=a.seed * 100000 + step, eos_tokens=eos)       # temperature 1.0: true samples of the policy
        comp, mask, ended = opd.completion_batch(out.tokens, a.max_new, IM_END, PAD)
        prompt_arr = jnp.asarray(np.asarray(out.padded_prompt_tokens)); comp_j = jnp.asarray(comp); mask_j = jnp.asarray(mask)
        t_sample = time.time() - t0

        t1 = time.time()                                                         # teacher reads the student's text
        t_lp = jnp.concatenate([common.compute_per_token_logps(t_gd, t_st, prompt_arr[i:i + MB], comp_j[i:i + MB],
                                                               pad_id=PAD, eos_id=IM_END) for i in range(0, BS, MB)])
        jax.block_until_ready(t_lp); t_teacher = time.time() - t1

        t2 = time.time()                                                         # student: gradient, then one update
        denom = jnp.maximum(mask_j.sum(), 1.0)
        g_acc, kl_sum = None, 0.0
        for i in range(0, BS, MB):
            sl = slice(i, i + MB)
            _, g, lp = grad_fn(params, rest, prompt_arr[sl], comp_j[sl], mask_j[sl], t_lp[sl], denom)
            g_acc = g if g_acc is None else jax.tree.map(jnp.add, g_acc, g)
            kl_sum += float(((lp - t_lp[sl]) * mask_j[sl]).sum())
        gnorm = float(optax.global_norm(g_acc))
        updates, opt_state = tx.update(g_acc, opt_state, params)
        params = optax.apply_updates(params, updates)
        nnx.update(student, params)                                              # in place: the sampler shares these variables
        jax.block_until_ready(params); t_update = time.time() - t2

        if step == 2:   # step 1 has learning rate 0 (warmup), so the check is only valid from step 2
            d_sampler = max(float(np.abs(np.array(opd.val(v)) - b).max()) for v, b in zip(lora_vars, before))
            d_params = max(float(np.abs(np.array(x) - x0).max()) for x, x0 in zip(jax.tree.leaves(params), params0))
            print(f"sync check at step 2: LoRA change in params = {d_params:.2e}, seen by the sampler = {d_sampler:.2e}")
            assert d_params > 0, "optimizer did not change the LoRA parameters"
            assert d_sampler > 0, "params changed but the sampler does not see them; stopping"

        n_valid = float(mask.sum()); gen = int(sum(len(t) for t in out.tokens))
        cum["gen"] += gen; cum["scored"] += int(n_valid); cum["sec"] += time.time() - t0
        rec = {"step": step, "gen_tokens": gen, "cum_gen_tokens": cum["gen"], "teacher_scored_tokens": int(n_valid),
               "cum_teacher_scored_tokens": cum["scored"], "mean_len": round(n_valid / BS, 1),
               "frac_ended": round(float(ended.mean()), 3), "reverse_kl": round(kl_sum / n_valid, 5),
               "grad_norm": round(gnorm, 4), "t_sample": round(t_sample, 1), "t_teacher": round(t_teacher, 1),
               "t_update": round(t_update, 1), "train_seconds": round(cum["sec"], 1)}
        open(f"{run}/metrics.jsonl", "a").write(json.dumps(rec) + "\n")
        print(f"step {step:3d} | rKL {rec['reverse_kl']:.4f} | len {rec['mean_len']:.0f} | ended {rec['frac_ended']:.2f} | "
              f"gnorm {gnorm:.3f} | sample {t_sample:.1f}s teacher {t_teacher:.1f}s update {t_update:.1f}s")
        if a.eval_every and step % a.eval_every == 0 and step < a.steps:
            do_eval(step)

    hbm("end of run")
    final = do_eval(a.steps)
    json.dump(final, open(f"{run}/final_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main(parse())
