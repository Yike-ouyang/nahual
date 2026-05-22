"""
This example uses a server within the environment defined on `https://github.com/afermg/EmbedSeg.git` (branch `nahual-wrap`).

Run `nix run github:afermg/EmbedSeg/nahual-wrap -- tcp://127.0.0.1:5125` from any
directory, or `nix develop --command bash -c "python server.py tcp://127.0.0.1:5125"`
from the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("embedseg")
address = "tcp://127.0.0.1:5125"

# %% Load model server-side. `weights` is required for real inference; without
# a checkpoint the model is randomly initialised and the output will be empty.
parameters = {
    "n_y": 256,
    "n_x": 256,
    "n_sigma": 2,
    "input_channels": 1,
    "num_classes": [4, 1],
    # optional
    # "weights": "/path/to/best_iou_model.pth",
    # "device": 0,
    # "seed_thresh": 0.9,
    # "fg_thresh": 0.5,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, single channel, single Z slice.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 1, 1, 256, 256)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (2, 256, 256) int32
