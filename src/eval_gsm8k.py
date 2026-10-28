import re

INSTR = "\n\nSolve step by step, then give the final answer as \\boxed{number}."

def load_gsm8k_test(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds.select(range(n))]

def make_prompt(hf_tok, q):
    return hf_tok.apply_chat_template([{"role": "user", "content": q + INSTR}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)

def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + len("\\boxed{")
    depth = 1
    while k < len(text) and depth:
        depth += (text[k] == "{") - (text[k] == "}")
        k += 1
    return text[j:k - 1] if depth == 0 else None

def first_number(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", "").replace("$", ""))
    return m.group(0) if m else None

def is_correct(pred, gold):
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False

def evaluate(sampler, hf_tok, eos, data, bs=16, max_new=384, max_prompt=256):
    import numpy as np
    prompts = [make_prompt(hf_tok, q) for q, _ in data]
    longest = max(len(hf_tok.encode(p)) for p in prompts)
    assert longest <= max_prompt, f"prompt of {longest} tokens exceeds max_prompt={max_prompt}"
    assert len(data) % bs == 0, "use a multiple of the batch size to avoid recompiling"
    correct = boxed = n_tok = 0
    width = None
    for i in range(0, len(data), bs):
        out = sampler(input_strings=prompts[i:i + bs], max_generation_steps=max_new,
                      max_prompt_length=max_prompt, temperature=0.0, eos_tokens=eos)
        if width is None:
            width = np.asarray(out.padded_prompt_tokens).shape
        for text, toks, (_, gold) in zip(out.text, out.tokens, data[i:i + bs]):
            b = last_boxed(text)
            boxed += b is not None
            correct += is_correct(first_number(b), gold)
            n_tok += len(toks)
    n = len(data)
    return {"n": n, "pass@1": round(correct / n, 4), "boxed_rate": round(boxed / n, 4),
            "mean_len": round(n_tok / n, 1), "prompt_batch_shape": list(width), "longest_prompt": longest}
