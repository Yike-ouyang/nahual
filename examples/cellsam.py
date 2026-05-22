"""
This example uses a server within the environment defined on `https://github.com/afermg/cellSAM.git`.

Run `nix run github:afermg/cellSAM -- tcp://127.0.0.1:5127` from any directory,
or `nix develop --command bash -c "python server.py tcp://127.0.0.1:5127"`
from the root of that repository. The default branch is the no-auth ONNX wrap;
the original DeepCell-auth path lives on the `nahual-wrap-deepcell` branch.

ONNX-only path — no DeepCell auth required. Weights are pulled from the
`keejkrej/cellsam-onnx` HuggingFace repo on first call.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("cellsam")
address = "tcp://127.0.0.1:5127"

# %% Load model server-side.
parameters = {
    # optional
    # "weights_dir": "/path/to/cached/onnx/weights",
    # "device": 0,
    # "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    # "bbox_threshold": 0.4,
    # "iou_threshold": 0.5,
    # "mask_threshold": 0.5,
    # "min_size": 25,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, float32.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, 512, 512)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (1, 512, 512) int32
