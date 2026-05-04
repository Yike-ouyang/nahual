"""
This example uses a server within the environment defined on `https://github.com/afermg/instanseg.git` (branch `nahual-wrap`).

Run `nix run github:afermg/instanseg/nahual-wrap -- ipc:///tmp/instanseg.ipc` from any
directory, or `nix develop --command bash -c "python server.py ipc:///tmp/instanseg.ipc"`
from the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("instanseg")
address = "ipc:///tmp/instanseg.ipc"

# %% Load model server-side. The default `brightfield_nuclei` model is fetched
# automatically on first call.
parameters = {
    "model_id": "brightfield_nuclei",
    # optional
    # "device": 0,
    # "expected_channels": 1,
}
response = setup(parameters, address=address)
print(response)

# %% Define custom data — NCZYX, float input.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((2, 1, 1, 512, 512)).astype(numpy.float32)
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (2, 512, 512) int32
