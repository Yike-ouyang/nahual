"""
This example uses a server within the environment defined on `https://github.com/afermg/stardist.git` (branch `nahual-wrap`).

Run `nix run github:afermg/stardist/nahual-wrap -- ipc:///tmp/stardist.ipc` from any
directory, or `nix develop --command bash -c "python server.py ipc:///tmp/stardist.ipc"`
from the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("stardist")
address = "ipc:///tmp/stardist.ipc"

# %% Load model server-side. All keys are optional; the empty dict gets you
# the default 2D fluorescence model on cuda:0.
parameters = {
    "model_name": "2D_versatile_fluo",
    # optional
    # "device": 0,
    # "expected_tile_size": 16,
    # "expected_channels": 1,
    # "prob_thresh": None,
    # "nms_thresh": None,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, single channel, single Z slice.
tile_size = 256
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 1, 1, tile_size, tile_size)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (2, 256, 256) int32
