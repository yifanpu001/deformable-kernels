import time
import torch
from deformable_kernels.modules import (
    CondConv2d,
    CondRotConv2d,
    CondRotConv2d_bmm,
    CondRotConv2d_forloop,
    CondRotConv2d_bmm_forloop,
)

Cin = 64
Cout = 64

conv_layer_CondConv2d = CondConv2d(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros')
conv_layer_CondRotConv2d = CondRotConv2d(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1)
conv_layer_CondRotConv2d_bmm = CondRotConv2d_bmm(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1)
conv_layer_CondRotConv2d_forloop = CondRotConv2d_forloop(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1)
conv_layer_CondRotConv2d_bmm_forloop = CondRotConv2d_bmm_forloop(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1)

input_tensor = torch.rand(64, Cin, 224, 224)
print(f'input_tensor.shape: {input_tensor.shape}')

start = time.time()
output_CondConv2d = conv_layer_CondConv2d(input_tensor)
end = time.time()
print(f"CondConv2d time: {end - start}")

start = time.time()
output_CondRotConv2d = conv_layer_CondRotConv2d(input_tensor)
end = time.time()
print(f"CondRotConv2d time: {end - start}")

start = time.time()
output_CondRotConv2d_bmm = conv_layer_CondRotConv2d_bmm(input_tensor)
end = time.time()
print(f"CondRotConv2d_bmm time: {end - start}")

start = time.time()
output_CondRotConv2d_forloop = conv_layer_CondRotConv2d_forloop(input_tensor)
end = time.time()
print(f"CondRotConv2d_forloop time: {end - start}")

start = time.time()
output_CondRotConv2d_bmm_forloop = conv_layer_CondRotConv2d_bmm_forloop(input_tensor)
end = time.time()
print(f"CondRotConv2d_bmm_forloop time: {end - start}")

print(f'CondConv2d                output shape: {output_CondConv2d.shape}')
print(f'CondRotConv2d             output shape: {output_CondRotConv2d.shape}')
print(f'CondRotConv2d_bmm         output shape: {output_CondRotConv2d_bmm.shape}')
print(f'CondRotConv2d_forloop     output shape: {output_CondRotConv2d_forloop.shape}')
print(f'CondRotConv2d_bmm_forloop output shape: {output_CondRotConv2d_bmm_forloop.shape}')

print("end")