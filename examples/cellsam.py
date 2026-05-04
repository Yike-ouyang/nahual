"""
This example uses a server within the environment defined on `https://github.com/afermg/cellSAM.git` (branch `nahual-wrap`).

Run `nix run github:afermg/cellSAM/nahual-wrap -- ipc:///tmp/cellsam.ipc` from any
directory, or `nix develop --command bash -c "python server.py ipc:///tmp/cellsam.ipc"`
from the root of that repository.

This is the original PyTorch path; it requires `DEEPCELL_ACCESS_TOKEN` to be set
in the server environment (https://users.deepcell.org). For an auth-free
alternative see `cellsam_onnx.py`.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("cellsam")
address = "ipc:///tmp/cellsam.ipc"

# %% Load model server-side.
parameters = {
    "model_name": "cellsam_general",
    # optional
    # "device": 0,
    # "version": None,
    # "bbox_threshold": 0.4,
    # "normalize": True,
    # "postprocess": False,
    # "remove_boundaries": False,
    # "fast": False,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, Z is squeezed (model operates in 2-D).
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, 512, 512)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (1, 512, 512) int32
