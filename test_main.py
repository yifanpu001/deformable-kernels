import time
import torch
from deformable_kernels.modules import CondConv2d, CondRotConv2d


input_tensor = torch.rand(64, 64, 224, 224)
print(f'input_tensor.shape: {input_tensor.shape}')

conv_layer_CondConv2d = CondConv2d(num_experts=4, in_channels=64, out_channels=64, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros')
conv_layer_CondRotConv2d = CondRotConv2d(num_experts=4, in_channels=64, out_channels=64, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1)

start = time.time()
output_CondConv2d = conv_layer_CondConv2d(input_tensor)
end = time.time()
print(f"CondConv2d time: {end - start}")

start = time.time()
output_CondRotConv2d = conv_layer_CondRotConv2d(input_tensor)
end = time.time()
print(f"CondRotConv2d time: {end - start}")

print(f'CondConv2d output shape: {output_CondConv2d.shape}')
print(f'CondRotConv2d output shape: {output_CondRotConv2d.shape}')

print("end")