"""GRPO baseline: the student samples G answers per question, reward = final answer correct (0/1),
advantage = group-normalised reward broadcast to every token, one policy-gradient update per batch.
No teacher and no KL penalty. Same LoRA setup, sampling and token budget as train_opd.py.

Colab:  %run scripts/train_grpo.py --name grpo_smoke --steps 20
"""
import argparse, json, os, subprocess, sys, time
try:
    import datasets
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

from src.models import make_mesh, hbm, load, STUDENT_ID   # imports src.env_setup first
from src.lora import apply_lora
from src import eval_gsm8k as ev
from src import opd

import numpy as np, optax, jax, jax.numpy as jnp
from flax import nnx
from datasets import load_dataset
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib, tokenizer_adapter


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=230)
    p.add_argument("--prompts", type=int, default=4, help="questions per step")
    p.add_argument("--group", type=int, default=4, help="answers per question; rollout batch = prompts x group")
    p.add_argument("--mb", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--max-prompt", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-n", type=int, default=192)
    p.add_argument("--ckpt-dir", default=None)
    return p.parse_args(argv)


def main(a):
    run = f"results/runs/{a.name}"; os.makedirs(run, exist_ok=False)   # a new name per run
    json.dump(vars(a), open(f"{run}/args.json", "w"), indent=2)
    from google.colab import drive; drive.mount("/content/drive")
    ckpt = a.ckpt_dir or f"/content/drive/MyDrive/opd-runs/{a.name}"; os.makedirs(ckpt, exist_ok=True)

    hf_tok = AutoTokenizer.from_pretrained(STUDENT_ID); tok = tokenizer_adapter.TokenizerAdapter(hf_tok)
    PAD = hf_tok.convert_tokens_to_ids("<|endoftext|>"); IM_END = hf_tok.convert_tokens_to_ids("<|im_end|>")
    eos = [IM_END, PAD]
    P, G, MB = a.prompts, a.group, a.mb; BS = P * G
    assert BS % MB == 0 and a.eval_n % BS == 0

    mesh = make_mesh()
    student = apply_lora(load(STUDENT_ID, mesh), mesh, seed=a.seed); hbm("student + LoRA")
    graphdef, params, rest = nnx.split(student, nnx.LoRAParam, ...)
    print(f"trainable LoRA params: {sum(int(np.prod(np.shape(x))) for x in jax.tree.leaves(params)) / 1e6:.2f}M")

    c = student.config
    smp = sampler_lib.Sampler(transformer=student, tokenizer=tok, cache_config=sampler_lib.CacheConfig(
        cache_size=a.max_prompt + a.max_new, num_layers=c.num_layers, num_kv_heads=c.num_kv_heads, head_dim=c.head_dim))
    lora_vars = [v for v in smp.model_def_and_state()[1] if isinstance(v, nnx.LoRAParam)]
    assert lora_vars, "the sampler holds no LoRA variables"
    before = [np.array(opd.val(v)) for v in lora_vars]
    opd.check_random_sampling(smp, ev.make_prompt(hf_tok, "What is 25 * 13? Show your work."), eos, a.max_prompt)
    params0 = [np.array(x) for x in jax.tree.leaves(params)]

    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, a.warmup, a.steps, a.lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched))
    opt_state = tx.init(params)
    grad_fn = opd.make_pg_grad_fn(graphdef, PAD, IM_END)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    order = np.random.RandomState(a.seed).permutation(len(ds))
    exam = ev.load_gsm8k_test(a.eval_n)
    cum = {"gen": 0, "sec": 0.0}

    def do_eval(step):
        t0 = time.time()
        res = ev.evaluate(smp, hf_tok, eos, exam, bs=BS, max_new=a.max_new, max_prompt=a.max_prompt)
        rec = {"step": step, "cum_gen_tokens": cum["gen"], "cum_teacher_scored_tokens": 0,
               "train_seconds": round(cum["sec"], 1), "eval_seconds": round(time.time() - t0), **res}
        open(f"{run}/eval.jsonl", "a").write(json.dumps(rec) + "\n"); print("EVAL", rec)
        opd.save_lora(params, f"{ckpt}/lora_step{step}.npz")
        return rec

    for step in range(1, a.steps + 1):
        t0 = time.time()
        rows = [ds[int(j)] for j in order[(step - 1) * P: step * P]]
        qs = [r["question"] for r in rows for _ in range(G)]
        golds = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows for _ in range(G)]
        prompts = [ev.make_prompt(hf_tok, q) for q in qs]
        out = smp(input_strings=prompts, max_generation_steps=a.max_new, max_prompt_length=a.max_prompt,
                  temperature=1.0, top_k=None, top_p=1.0, seed=a.seed * 100000 + step, eos_tokens=eos)
        comp, mask, ended = opd.completion_batch(out.tokens, a.max_new, IM_END, PAD)
        prompt_arr = jnp.asarray(np.asarray(out.padded_prompt_tokens)); comp_j = jnp.asarray(comp); mask_j = jnp.asarray(mask)

        correct = np.array([float(ev.is_correct(ev.first_number(ev.last_boxed(t)), g)) for t, g in zip(out.text, golds)])
        rewards = correct * ended.astype(np.float64)   # an answer that never stops earns nothing
        R = rewards.reshape(P, G)
        adv_seq = ((R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-4)).reshape(-1)   # 0 for groups with no signal
        adv = jnp.asarray(np.repeat(adv_seq[:, None], comp.shape[1], axis=1).astype(np.float32))
        t_sample = time.time() - t0

        t2 = time.time()
        denom = jnp.maximum(mask_j.sum(), 1.0)
        g_acc = None
        for i in range(0, BS, MB):
            sl = slice(i, i + MB)
            _, g, _ = grad_fn(params, rest, prompt_arr[sl], comp_j[sl], mask_j[sl], adv[sl], denom)
            g_acc = g if g_acc is None else jax.tree.map(jnp.add, g_acc, g)
        gnorm = float(optax.global_norm(g_acc))
        updates, opt_state = tx.update(g_acc, opt_state, params)
        params = optax.apply_updates(params, updates)
        nnx.update(student, params)
        jax.block_until_ready(params); t_update = time.time() - t2

        if step == 3:   # learning rate is 0 at step 1, and step-2 gradients can be zero when no group has signal
            d_sampler = max(float(np.abs(np.array(opd.val(v)) - b).max()) for v, b in zip(lora_vars, before))
            d_params = max(float(np.abs(np.array(x) - x0).max()) for x, x0 in zip(jax.tree.leaves(params), params0))
            print(f"sync check at step 3: LoRA change in params = {d_params:.2e}, seen by the sampler = {d_sampler:.2e}")
            if d_params > 0: assert d_sampler > 0, "params changed but the sampler does not see them; stopping"

        gen = int(sum(len(t) for t in out.tokens)); n_valid = float(mask.sum())
        cum["gen"] += gen; cum["sec"] += time.time() - t0
        rec = {"step": step, "gen_tokens": gen, "cum_gen_tokens": cum["gen"], "mean_len": round(n_valid / BS, 1),
               "frac_ended": round(float(ended.mean()), 3), "reward_mean": round(float(rewards.mean()), 3), "correct_mean": round(float(correct.mean()), 3),
               "groups_with_signal": round(float((R.std(1) > 0).mean()), 2), "grad_norm": round(gnorm, 4),
               "t_sample": round(t_sample, 1), "t_update": round(t_update, 1), "train_seconds": round(cum["sec"], 1)}
        open(f"{run}/metrics.jsonl", "a").write(json.dumps(rec) + "\n")
        print(f"step {step:3d} | reward {rec['reward_mean']:.2f} | signal {rec['groups_with_signal']:.2f} | len {rec['mean_len']:.0f} | "
              f"ended {rec['frac_ended']:.2f} | gnorm {gnorm:.3f} | sample {t_sample:.1f}s update {t_update:.1f}s")
        if a.eval_every and step % a.eval_every == 0 and step < a.steps:
            do_eval(step)

    hbm("end of run")
    final = do_eval(a.steps)
    json.dump(final, open(f"{run}/final_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main(parse())
