"""LoRA setup shared by SFT, GRPO and on-policy distillation, so all three train the same parameters."""
import jax
from flax import nnx
import qwix

LORA_MODULES = ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"

def apply_lora(base, mesh, rank=16, alpha=32.0, seed=None):
    """Wrap a Tunix model with LoRA on all seven projections. seed=None reproduces the first SFT run."""
    provider = qwix.LoraProvider(module_path=LORA_MODULES, rank=rank, alpha=alpha)
    extra = {} if seed is None else {"rngs": nnx.Rngs(seed)}
    lora = qwix.apply_lora_to_model(base, provider, **extra, **base.get_model_input())
    with mesh:
        st = nnx.state(lora)
        nnx.update(lora, jax.lax.with_sharding_constraint(st, nnx.get_partition_spec(st)))
    return lora
