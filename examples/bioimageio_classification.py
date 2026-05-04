"""
This example uses a server within the environment defined on `https://github.com/afermg/nahual_bioimageio.git`.

Run `nix run github:afermg/nahual_bioimageio -- ipc:///tmp/bioimageio.ipc` from any
directory. The default variant (ONNX/TorchScript) is enough for `polite-pig`.

Task: Human Protein Atlas (HPA) sub-cellular protein-localisation classification.
Output is a per-class score vector, not a segmentation mask. This is a task type
not covered by any other Nahual wrap (no other classifier).
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("bioimageio")
address = "ipc:///tmp/bioimageio.ipc"

# %% polite-pig: HPA protein-localisation classifier. bcyx, 4 channels at 512².
parameters = {
    "source": "polite-pig",
    # optional
    # "weight_format": "torchscript",
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)

# %% bcyx with C=4 (the four HPA channels: protein, microtubule, ER, nucleus).
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 4, 512, 512)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: per-class probability vector (e.g. (1, 19) for the HPA label set).
