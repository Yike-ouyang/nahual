"""
This example uses a server within the environment defined on `https://github.com/afermg/nahual_vit.git`.

Run `nix develop --command bash -c "python server.py tcp://127.0.0.1:5120"` from the root directory of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("vit")
address = "tcp://127.0.0.1:5120"
"""
Run Visual Transformers using the HuggingFace's transformers library.

Run `nix develop --command bash -c "python src/vit/server.py tcp://127.0.0.1:5120"` from the root directory of https://github.com/afermg/nahual_vit.
"""

# %%Load models server-side
parameters = dict(model_name="recursionpharma/OpenPhenom")
response = setup(parameters, address=address)

# %% Define custom data
tile_size = 256

# channel can be < 6 and the model will pad
input_shape = (2, 6, 1, tile_size, tile_size)
data = numpy.random.random_sample(input_shape)
result = process(data, address=address)
