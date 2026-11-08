"""On-policy distillation core: stop-token handling, log-probs, and the gradient function."""
import jax, jax.numpy as jnp, numpy as np, optax
from flax import nnx
from tunix.rl import common

def val(v):
    """Read a Variable's array across Flax versions."""
    return v.get_value() if hasattr(v, "get_value") else v.value

def completion_batch(tokens_list, max_new, im_end, pad):
    """Sampler output -> padded completions [B, max_new+1], loss mask, and a finished-on-its-own flag.
    The sampler strips the stop token, so answers shorter than the cap get it appended before scoring."""
    B, C = len(tokens_list), max_new + 1
    comp = np.full((B, C), pad, np.int32); mask = np.zeros((B, C), np.float32); ended = np.zeros(B, bool)
    for i, t in enumerate(tokens_list):
        t = [int(x) for x in t]
        e = len(t) < max_new
        ids = t + ([im_end] if e else [])
        comp[i, :len(ids)] = ids; mask[i, :len(ids)] = 1.0; ended[i] = e
    return comp, mask, ended

def token_logps(model, prompt, comp, pad_id, eos_id):
    """Per-token log-prob of the completion tokens (same alignment as tunix common.compute_per_token_logps)."""
    input_tokens, positions, attn_mask, _ = common.process_ids(prompt, comp, pad_id, eos_id)
    logits, _ = model(input_tokens, positions, None, attn_mask)
    C = comp.shape[1]
    lg = logits[:, -C - 1:-1, :].astype(jnp.float32)
    return -optax.softmax_cross_entropy_with_integer_labels(lg, comp)

def make_grad_fn(graphdef, pad_id, eos_id):
    @jax.jit
    def grad_fn(params, rest, prompt, comp, mask, teacher_lp, denom):
        def loss_fn(p):
            model = nnx.merge(graphdef, p, rest)
            lp = token_logps(model, prompt, comp, pad_id, eos_id)
            adv = jax.lax.stop_gradient(teacher_lp - lp)        # advantage: no gradient through it
            return -(adv * lp * mask).sum() / denom, lp
        (loss, lp), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, g, jax.lax.stop_gradient(lp)
    return grad_fn


def save_lora(params, path):
    """Save LoRA params to .npz (bf16 stored as float32, which is lossless)."""
    np.savez(path, **{f"p{i}": np.asarray(x).astype(np.float32) for i, x in enumerate(jax.tree.leaves(params))})

def load_lora(template, path):
    z = np.load(path); leaves, treedef = jax.tree.flatten(template)
    new = [jnp.asarray(z[f"p{i}"]).astype(x.dtype) for i, x in enumerate(leaves)]
    assert len(z.files) == len(leaves) and all(n.shape == x.shape for n, x in zip(new, leaves)), "checkpoint does not match this LoRA setup"
    return jax.tree.unflatten(treedef, new)

def make_pg_grad_fn(graphdef, pad_id, eos_id):
    """REINFORCE-style gradient with externally supplied per-token advantages (GRPO)."""
    @jax.jit
    def grad_fn(params, rest, prompt, comp, mask, adv, denom):
        def loss_fn(p):
            model = nnx.merge(graphdef, p, rest)
            lp = token_logps(model, prompt, comp, pad_id, eos_id)
            return -(adv * lp * mask).sum() / denom, lp
        (loss, lp), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, g, jax.lax.stop_gradient(lp)
    return grad_fn


def check_random_sampling(smp, prompt_text, eos, max_prompt=256):
    """Identical prompts must yield different samples; Tunix 0.1.7 silently decodes greedily without top_p."""
    o = smp(input_strings=[prompt_text] * 4, max_generation_steps=32, max_prompt_length=max_prompt,
            temperature=1.0, top_k=None, top_p=1.0, seed=12345, eos_tokens=eos)
    n = len({tuple(int(x) for x in t) for t in o.tokens})
    assert n > 1, "sampler returned identical outputs for identical prompts: rollouts are not random"
    print(f"sampling check: {n}/4 distinct outputs for identical prompts")
