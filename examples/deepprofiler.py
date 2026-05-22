"""
This example uses a server within the environment defined on `https://github.com/afermg/DeepProfiler.git`.

Run `nix run --impure github:afermg/DeepProfiler -- tcp://127.0.0.1:5113`
from the root directory of that repository (or `nix develop --impure --command
python server.py tcp://127.0.0.1:5113`).

Note: DeepProfiler's CPCNNv1 backbone is a Keras ResNet50V2 (TF 2.13 + tf-keras
2.17 in legacy mode) that emits a 2048-d feature embedding per single-cell crop.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("deepprofiler")
address = "tcp://127.0.0.1:5113"

# %% Load model server-side (server defaults: ResNet50V2 + ImageNet weights).
parameters = {
    # optional
    # "backbone": "resnet50v2",
    # "weights": "imagenet",  # or a path to a .h5 checkpoint, or None
    # "expected_channels": 3,
    # "input_size": 224,
    # "device": 0,  # None → GPU:0 if visible, else CPU
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, with C=3 (RGB) and Z=1.
tile_size = 224
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 3, 1, tile_size, tile_size))
result = process(data, address=address)
print(result.shape)
# Expected: (2, 2048)
