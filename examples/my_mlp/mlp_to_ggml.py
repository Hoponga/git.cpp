# Convert Pytorch MLP model to GGML format 

import torch 
import struct
import numpy as np

# load model 
model = torch.load("mnist_model.pth", weights_only = False)

model_state_dict = model

output_stream = open("./data/model.bin", "wb")

for name in model_state_dict.keys():
    data = model_state_dict[name].squeeze().numpy()
    # number of dimensions
    num_dims = len(data.shape)
    output_stream.write(struct.pack("i", num_dims))
    # dimension length and data
    data = data.astype(np.float32)
    for i in range(num_dims):
        output_stream.write(struct.pack("i", data.shape[num_dims - 1 - i]))
    data.tofile(output_stream)

output_stream.close()

