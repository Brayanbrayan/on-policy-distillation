from src import env_setup  # noqa: F401  (must precede any torch import)
import jax
from flax import nnx
from tunix.models.automodel import AutoModel

STUDENT_ID = "Qwen/Qwen3-0.6B"
TEACHER_ID = "Qwen/Qwen3-1.7B"

def make_mesh():
    return jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)

def hbm(tag):
    s = jax.devices()[0].memory_stats()
    print(f"{tag}: in use {s['bytes_in_use']/2**30:.2f} GiB | peak {s['peak_bytes_in_use']/2**30:.2f} | limit {s['bytes_limit']/2**30:.2f}")

def describe(model):
    leaves = jax.tree_util.tree_leaves(nnx.state(model))
    n = sum(getattr(l, "size", 0) for l in leaves)
    print(f"  params: {n/1e9:.2f}B | dtypes: {sorted({str(getattr(l, 'dtype', '?')) for l in leaves})}")

def load(model_id, mesh, root="/content/models"):
    path = f"{root}/{model_id.split('/')[-1].lower()}"
    model, _ = AutoModel.from_pretrained(model_id=model_id, mesh=mesh, model_download_path=path)
    return model
