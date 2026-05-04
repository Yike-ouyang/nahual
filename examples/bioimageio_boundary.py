"""
This example uses a server within the environment defined on `https://github.com/afermg/nahual_bioimageio.git`.

Run `nix run github:afermg/nahual_bioimageio -- ipc:///tmp/bioimageio.ipc` from any
directory. The default variant covers `hiding-blowfish`.

Task: 2-D EM mitochondria boundary / contour detection. Output is a continuous
boundary probability map, not an instance-label map — a different output type
than any of the segmentation wraps in this repo.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("bioimageio")
address = "ipc:///tmp/bioimageio.ipc"

# %% hiding-blowfish: 2-D EM mitochondria boundary detection. bcyx input.
parameters = {
    "source": "hiding-blowfish",
    # optional
    # "weight_format": "torchscript",
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)

# %% Single-channel EM-style image at 256².
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 256, 256)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (1, 1, 256, 256) float — boundary probability map (0..1).
