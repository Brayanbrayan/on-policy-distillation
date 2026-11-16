"""Sample teacher answers on GSM8K training questions for SFT.
Colab:  %run scripts/gen_teacher_data.py --name sft_data_t1 --out /content/drive/MyDrive/opd-runs/sft_data_t1/teacher_samples.jsonl
Resumes if the output file exists (whole batches are appended). Does not touch git; call save() afterwards."""
import argparse, json, os, random, subprocess, sys, time
try:
    import datasets
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)
from src.models import make_mesh, hbm, load, TEACHER_ID
from src import eval_gsm8k as ev
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib, tokenizer_adapter


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--out", required=True)
    p.add_argument("--token-budget", type=int, default=1_000_000)
    p.add_argument("--temperature", type=float, default=1.0, help="0 = greedy")
    p.add_argument("--bs", type=int, default=32); p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--max-prompt", type=int, default=256); p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(a):
    from google.colab import drive; drive.mount("/content/drive")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(ds))); random.Random(0).shuffle(order)
    hf_tok = AutoTokenizer.from_pretrained(TEACHER_ID); tok = tokenizer_adapter.TokenizerAdapter(hf_tok)
    IM_END = hf_tok.convert_tokens_to_ids("<|im_end|>"); eos = [IM_END, hf_tok.convert_tokens_to_ids("<|endoftext|>")]
    teacher = load(TEACHER_ID, make_mesh()); hbm("teacher loaded"); c = teacher.config
    smp = sampler_lib.Sampler(transformer=teacher, tokenizer=tok, cache_config=sampler_lib.CacheConfig(
        cache_size=a.max_prompt + a.max_new, num_layers=c.num_layers, num_kv_heads=c.num_kv_heads, head_dim=c.head_dim))
    kw = dict(temperature=0.0) if a.temperature == 0 else dict(temperature=a.temperature, top_k=None, top_p=1.0)

    done = [json.loads(l) for l in open(a.out)] if os.path.exists(a.out) else []
    i, tokens_done, t0 = len(done), sum(r["n_gen"] for r in done), time.time()
    print(f"resuming at {i} samples, {tokens_done} tokens")
    while tokens_done < a.token_budget and i + a.bs <= len(order):
        rows = [ds[int(j)] for j in order[i:i + a.bs]]
        prompts = [ev.make_prompt(hf_tok, r["question"]) for r in rows]
        extra = {} if a.temperature == 0 else {"seed": a.seed * 100000 + i // a.bs}
        out = smp(input_strings=prompts, max_generation_steps=a.max_new, max_prompt_length=a.max_prompt,
                  eos_tokens=eos, **kw, **extra)
        lines = []
        for r, p, text, toks in zip(rows, prompts, out.text, out.tokens):
            toks = [int(t) for t in toks]; ended = len(toks) < a.max_new
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            lines.append(json.dumps({"prompt": p, "gold": gold, "text": text, "n_gen": len(toks), "ended": ended,
                                     "correct": bool(ev.is_correct(ev.first_number(ev.last_boxed(text)), gold)),
                                     "completion_ids": toks + ([IM_END] if ended else [])}))
            tokens_done += len(toks)
        with open(a.out, "a") as f: f.write("\n".join(lines) + "\n")
        i += a.bs
        if (i // a.bs) % 10 == 0:
            print(f"{i} samples | {tokens_done:,} tokens | {tokens_done / (time.time() - t0):.0f} tok/s since start")

    rows = [json.loads(l) for l in open(a.out)]
    stats = {"name": a.name, "temperature": a.temperature, "top_p": 1.0 if a.temperature else None, "n_samples": len(rows),
             "tokens": sum(r["n_gen"] for r in rows), "mean_len": round(float(np.mean([r["n_gen"] for r in rows])), 1),
             "truncated_frac": round(1 - float(np.mean([r["ended"] for r in rows])), 4),
             "teacher_sample_acc": round(float(np.mean([r["correct"] for r in rows])), 4)}
    print(stats); os.makedirs("results/data_stats", exist_ok=True)
    json.dump(stats, open(f"results/data_stats/{a.name}.json", "w"), indent=2)


if __name__ == "__main__":
    main(parse())
