import os
import time
import torch
from deformable_kernels.modules import (
    CondConv2d,
    CondRotConv2d,
    CondRotConv2d_bmm,
    CondRotConv2d_forloop,
    CondRotConv2d_bmm_forloop,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'  # '0, 1, 2, 3'
Cin = 64
Cout = 64

conv_layer_CondConv2d = CondConv2d(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros').cuda()
conv_layer_CondRotConv2d = CondRotConv2d(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1).cuda()
conv_layer_CondRotConv2d_bmm = CondRotConv2d_bmm(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1).cuda()
conv_layer_CondRotConv2d_forloop = CondRotConv2d_forloop(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1).cuda()
conv_layer_CondRotConv2d_bmm_forloop = CondRotConv2d_bmm_forloop(num_experts=4, in_channels=Cin, out_channels=Cout, kernel_size=3, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False,
                                   padding_mode='zeros', proportion=1).cuda()

L1, L2, L3, L4, L5 = [], [], [], [], []

for idx in range(20):
    print(f'{idx}  ', end='')
    if (idx + 1) % 20 == 0:
        print('')
    input_tensor = torch.rand(64, Cin, 224, 224).cuda()
    # print(f'input_tensor.shape: {input_tensor.shape}')

    start = time.time()
    output_CondConv2d = conv_layer_CondConv2d(input_tensor)
    end = time.time()
    L1.append(end - start)

    start = time.time()
    output_CondRotConv2d = conv_layer_CondRotConv2d(input_tensor)
    end = time.time()
    L2.append(end - start)

    start = time.time()
    output_CondRotConv2d_bmm = conv_layer_CondRotConv2d_bmm(input_tensor)
    end = time.time()
    L3.append(end - start)

    start = time.time()
    output_CondRotConv2d_forloop = conv_layer_CondRotConv2d_forloop(input_tensor)
    end = time.time()
    L4.append(end - start)

    start = time.time()
    output_CondRotConv2d_bmm_forloop = conv_layer_CondRotConv2d_bmm_forloop(input_tensor)
    end = time.time()
    L5.append(end - start)
print('')
print(f"CondConv2d                time: {sum(L1) / len(L1)}")
print(f"CondRotConv2d             time: {sum(L2) / len(L2)}")
print(f"CondRotConv2d_bmm         time: {sum(L3) / len(L3)}")
print(f"CondRotConv2d_forloop     time: {sum(L4) / len(L4)}")
print(f"CondRotConv2d_bmm_forloop time: {sum(L5) / len(L5)}")

# print(f'CondConv2d                output shape: {output_CondConv2d.shape}')
# print(f'CondRotConv2d             output shape: {output_CondRotConv2d.shape}')
# print(f'CondRotConv2d_bmm         output shape: {output_CondRotConv2d_bmm.shape}')
# print(f'CondRotConv2d_forloop     output shape: {output_CondRotConv2d_forloop.shape}')
# print(f'CondRotConv2d_bmm_forloop output shape: {output_CondRotConv2d_bmm_forloop.shape}')

print("end")

"""
result(GPU, this version):
0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  
20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  
40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  
60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  
80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99  
CondConv2d                time: 0.005925219058990478
CondRotConv2d             time: 0.38435002326965334
CondRotConv2d_bmm         time: 0.382738881111145
CondRotConv2d_forloop     time: 0.21280317068099974
CondRotConv2d_bmm_forloop time: 0.222123601436615
end

result(CPU, not this code):
0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  
20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  
40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  
60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  
80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99  
100  101  102  103  104  105  106  107  108  109  110  111  112  113  114  115  116  117  118  119  
120  121  122  123  124  125  126  127  128  129  130  131  132  133  134  135  136  137  138  139  
140  141  142  143  144  145  146  147  148  149  150  151  152  153  154  155  156  157  158  159  
160  161  162  163  164  165  166  167  168  169  170  171  172  173  174  175  176  177  178  179  
180  181  182  183  184  185  186  187  188  189  190  191  192  193  194  195  196  197  198  199  
CondConv2d                time: 0.5828519403934479
CondRotConv2d             time: 0.67635049700737
CondRotConv2d_bmm         time: 0.6552649986743927
CondRotConv2d_forloop     time: 0.6211545383930206
CondRotConv2d_bmm_forloop time: 0.7428855657577514
end

result(GPU, device=weights.device)
卡不空，还没测。
"""