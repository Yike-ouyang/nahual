"""
This client matches https://github.com/afermg/ultrack/blob/nahual-wrap/server.py

Run `nix run github:afermg/ultrack -- tcp://127.0.0.1:5124` from any
directory, or `nix develop --command bash -c "python server.py
tcp://127.0.0.1:5124"` from the root of that repository.

Ultrack is a *tracking* server: input is a per-frame foreground/contour stack
shaped (T, 2, Z, Y, X), output is an integer label volume of track IDs
shaped (T, Z, Y, X) (3-D) or (T, Y, X) (2-D).
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("ultrack")
address = "tcp://127.0.0.1:5124"

# %% Load model server-side. All keys are optional; the empty dict gets you
# ultrack defaults.
parameters = {
    "segmentation": {"min_area": 50, "max_area": 50_000},
    "linking": {"max_distance": 25.0, "max_neighbors": 5},
    "tracking": {
        "appear_weight": -0.001,
        "disappear_weight": -0.001,
        "division_weight": -1.0,
        "n_threads": 1,
    },
}
response = setup(parameters, address=address)
print(response)

# %% Synthesize a tiny 2-D timelapse (T=4, C=2, Z=1, Y=128, X=128).
#    Channel 0 = foreground probability map, channel 1 = contour map.
n_t, size = 4, 128
yy, xx = numpy.meshgrid(numpy.arange(size), numpy.arange(size), indexing="ij")
data = numpy.zeros((n_t, 2, 1, size, size), dtype=numpy.float32)
for t in range(n_t):
    cy, cx, r = size // 2 + 2 * t, size // 2 + 2 * t, 18.0
    dist = numpy.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    data[t, 0, 0] = numpy.clip(1.0 - dist / r, 0.0, 1.0)
    data[t, 1, 0] = numpy.exp(-((dist - r) ** 2) / (2.0 * 2.0**2))

# %% Track. Result is (T, Y, X) of int track IDs.
result = process(data, address=address)
print(f"shape={result.shape} dtype={result.dtype}")
print(f"unique track ids: {sorted(numpy.unique(result).tolist())[:20]}")
