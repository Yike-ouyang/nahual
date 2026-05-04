"""
This example uses a server within the environment defined on `https://github.com/afermg/nahual_bioimageio.git`.

Run `nix run github:afermg/nahual_bioimageio#with-careamics -- ipc:///tmp/bioimageio.ipc`
from any directory. The `with-careamics` variant pulls in the CAREamics backend
needed by Noise2Void / N2V2 / CARE-style restoration models.

Task: 2-D image denoising (Noise2Void). Output shape matches the input — a
denoised replica, not a label map.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("bioimageio")
address = "ipc:///tmp/bioimageio.ipc"

# %% jolly-ox: Noise2Void 2-D denoiser. bcyx input, bcyx output (same shape).
parameters = {
    "source": "jolly-ox",
    # optional
    # "weight_format": "pytorch_state_dict",
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)

# %% Synthesise a noisy 1-channel image. bcyx layout — match RDF input axes.
numpy.random.seed(seed=42)
clean = numpy.zeros((1, 1, 128, 128), dtype=numpy.float32)
yy, xx = numpy.meshgrid(numpy.arange(128), numpy.arange(128), indexing="ij")
clean[0, 0] = numpy.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 20.0**2))
data = clean + 0.2 * numpy.random.standard_normal(clean.shape).astype(numpy.float32)

result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
# Expected: (1, 1, 128, 128) float32 — a denoised version of the input.
