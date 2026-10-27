import os
# Must run before anything imports torch. Colab's preinstalled torch-tpu backend
# crashes against the libtpu version Tunix pins. We never run torch on the TPU.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
