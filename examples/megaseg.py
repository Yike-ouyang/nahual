"""
This example uses a server within the environment defined on `https://github.com/afermg/allencell-segmenter-ml.git` (branch `nahual-wrap`).

Run `nix run github:afermg/allencell-segmenter-ml/nahual-wrap -- ipc:///tmp/megaseg.ipc`
from any directory, or `nix develop --command bash -c "python server.py ipc:///tmp/megaseg.ipc"`
from the root of that repository.

MegaSeg is the Allen Institute MegaSegmenter — 3-D, single-channel input, returns
binary masks shaped (N, 1, Z, Y, X).
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("megaseg")
address = "ipc:///tmp/megaseg.ipc"

# %% Load model server-side. Without `checkpoint_path` the server uses a
# bundled MegaSeg checkpoint.
parameters = {
    # optional
    # "checkpoint_path": "/path/to/megaseg.ckpt",
    # "cache_dir": "/tmp/megaseg-cache",
    # "device": 0,
    # "overlap": 0.0,
    # "sw_batch_size": 1,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, single channel, small 3-D volume.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 16, 128, 128)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (1, 1, 16, 128, 128) uint8 — binary mask
