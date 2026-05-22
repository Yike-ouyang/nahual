"""
This example uses a server within the environment defined on `https://github.com/afermg/micro-sam.git` (branch `nahual-wrap`).

Run `nix run --impure github:afermg/micro-sam/nahual-wrap -- tcp://127.0.0.1:5122`
from any directory, or `nix develop --impure --command bash -c "python server.py tcp://127.0.0.1:5122"`
from the root of that repository.

Cold-cache build is ~30 min — pre-warm with `nix develop --impure --command true`.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("microsam")
address = "tcp://127.0.0.1:5122"

# %% Load model server-side. `vit_b_lm` is the light-microscopy ViT-B variant.
parameters = {
    "model_type": "vit_b_lm",
    # optional
    # "device": 0,
    # "checkpoint": "/path/to/checkpoint.pt",
    # "segmentation_mode": "auto",
    # "is_tiled": False,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, Z must be 1 (singleton).
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, 512, 512)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (1, 512, 512) int32
