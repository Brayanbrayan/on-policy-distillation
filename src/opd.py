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
