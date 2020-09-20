import os

import torch
from torchviz import make_dot

from Model_define_pytorch import AutoEncoder

# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)
# parameters for data
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

# Model construction
model = AutoEncoder(feedback_bits)
if use_single_gpu:
    model = model.cuda()

else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    autoencoder = torch.nn.DataParallel(model).cuda()

print('Drawing model ...')
test_data = torch.rand(1, img_channels, img_height, img_width).cuda()
y = model(test_data)
g = make_dot(y)
g.render('Linear-Convolutional_model', view=False)

print('Graph drawn!')
