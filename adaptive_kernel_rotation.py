import math
import torch
import matplotlib.pyplot as plt 


"""
We only consider Bilinear 3x3 resample scenario.

The last kernel rotation function only rotates a batch of Cout x Cin x k x k kernels with same degree.

In this function, we will rotation num_experts Cout x Cin x k x k kernels with different degrees respectively.
"""

def rotate_3x3_kernel_adaptive_forloop(weights, num_experts, kernel_theta_list):
    """
    Args:
        weights: tensor, shape = [num_experts, Cout, Cin, k, k]
        num_experts: number of experts
        kernel_theta: a list of float number with the size of num_experts
    """
    assert(weights.shape[3] == 3)
    assert(weights.shape[4] == 3)
    assert(max(kernel_theta_list) <= 45.0)
    assert(min(kernel_theta_list) >= - 45.0)

    for idx in range(num_experts):

        kernel_theta = kernel_theta_list[idx]
        weight = weights[idx]

        is_clockwise = kernel_theta < 0
        kernel_theta = - kernel_theta if is_clockwise else kernel_theta 

        x = math.cos(kernel_theta / 180. * math.pi)
        y = math.sin(kernel_theta / 180. * math.pi)

        a = x - y
        b = x * y
        c = x + y

        Alpha = torch.tensor([[ a, 1-a, 0.,  0.,     0., 0.,  0.,  0., 0.],           # w'(-1, 1)
                              [0., x-b,  b,  0., 1-c+b, y-b,  0.,  0., 0.],           # w'( 0, 1)
                              [0.,  0.,  a,  0.,    0., 1-a,  0.,  0., 0.],           # w'( 1, 1)
                              [ b, y-b, 0., x-b, 1-c+b,  0.,  0.,  0., 0.],           # w'(-1, 0)
                              [0.,  0., 0.,  0.,    1.,  0.,  0.,  0., 0.],           # w'( 0, 0)
                              [0.,  0., 0.,  0., 1-c+b, x-b,  0., y-b,  b],           # w'( 1, 0)
                              [0.,  0., 0., 1-a,    0.,  0.,   a,  0., 0.],           # w'(-1,-1)
                              [0.,  0., 0., y-b, 1-c+b,  0.,   b, x-b, 0.],           # w'( 0,-1)
                              [0.,  0., 0.,  0.,    0.,  0.,  0., 1-a,  a]])   # w'( 1,-1).cuda()

        Cout, Cin, _, _ = weight.shape

        if is_clockwise:
            weight = weight.transpose(2, 3).contiguous().view(Cout * Cin, 9)
            weight = weight.transpose(0, 1)
            weight = (torch.mm(Alpha, weight).transpose(0, 1)).view(Cout, Cin, 3, 3).transpose(2, 3)
        else:
            weight = weight.view(Cout * Cin, 9)
            weight = weight.transpose(0, 1)
            weight = (torch.mm(Alpha, weight).transpose(0, 1)).view(Cout, Cin, 3, 3)


        weights[idx] = weight

    return weights


def plotshow(input, i=0, j=0, name='before'):
    plt.matshow(input[i][j].numpy())
    plt.show()
    plt.savefig(f'{name}.png')


def augmentKernel(weight, flip_dim=None,rotate_mode = 'bilinear', kernel_theta = 0., sigma = 0.1):
    # if kernel_theta == 0.:
    #     return weight
    # assert(weight.shape[2]==weight.shape[3]==3)
    # assert(-45. <= kernel_theta <= 45.)
    # if flip_dim is not None:
    #     assert(set(flip_dim).issubset(set([2,3])))
    #     weight = torch.flip(weight,flip_dim)
    is_clockwise = kernel_theta > 0
    kernel_theta = kernel_theta if is_clockwise else -kernel_theta 

    x = math.cos(kernel_theta / 180. * math.pi)
    y = math.sin(kernel_theta / 180. * math.pi)
    
    if rotate_mode == 'gaussian':
        if is_clockwise:
            T = torch.tensor([[x, y],
                            [-y, x]])
        else:
            T = torch.tensor([[x, -y],
                            [y, x]])
        A = torch.tensor([[-1.,-1.],[-1.,0.],[-1.,1.],
                        [0,-1],[0,0],[0,1],
                        [1,-1],[1,0],[1,1]])
        A = A.transpose(0,1)    # A: 2x9
        B = torch.mm(T,A)       # B: 2x9
        # print(A)
        # print(B)
        C = torch.mm(A.transpose(0,1),B) # C: 9x9
        A_sq = (torch.sum(A**2,0).view(9,1)).repeat(1,9) # 9x1 -> 9x9
        B_sq = (torch.sum(B**2,0).view(1,9)).repeat(9,1) # 1x9 -> 9x9
        Alpha = A_sq + B_sq -2*C

        # Beta = torch.zeros(9,9)
        # for i in range(9):
        #     for j in range(9):
        #         Beta[i,j] = torch.sum((A[:,i]-B[:,j])**2)
        # print(Alpha - Beta)
        Alpha = torch.exp(-Alpha / sigma) # alpha: 9x9
        Alpha = Alpha / torch.sum(Alpha,1).view(9,1)
        Alpha[4,:] = torch.tensor([0.,0,0,0,1,0,0,0,0])
        inp_c, out_c, _,_ = weight.shape
        weight = weight.view(inp_c*out_c,9)
        weight = weight.transpose(0,1)
        weight = (torch.mm(Alpha.cuda(),weight).transpose(0,1)).view(inp_c,out_c,3,3)
    else:  # Bilinear
        a = x-y
        b = x*y
        c = x+y

        Alpha = torch.tensor([[a, 1-a, 0.,0.,0.,0.,0.,0.,0.],
                            [0.,x-b,b,0.,1-c+b,y-b,0.,0.,0.],
                            [0.,0.0,a,0.,0.0,1-a,0.,0.,0.],
                            [b,y-b,0.,x-b,1-c+b,0.,0.,0.,0.],
                            [0.,0.,0.,0.,1.,0.,0.,0.,0.],
                            [0.,0.,0.,0.,1-c+b,x-b,0.,y-b,b],
                            [0.,0.,0.,1-a,0.,0.,a,0.,0.],
                            [0.,0.,0.,y-b,1-c+b,0.,b,x-b,0.],
                            [0.,0.,0.,0.,0.,0.,0.,1-a,a]]).cuda()
        inp_c, out_c, _,_ = weight.shape
        inp_c, out_c, _,_ = weight.shape
        if is_clockwise:
            weight = weight.transpose(2,3).contiguous().view(inp_c*out_c,9)
            weight = weight.transpose(0,1)
            weight = (torch.mm(Alpha,weight).transpose(0,1)).view(inp_c,out_c,3,3).transpose(2,3)
        else:
            weight = weight.view(inp_c*out_c,9)
            weight = weight.transpose(0,1)
            weight = (torch.mm(Alpha,weight).transpose(0,1)).view(inp_c,out_c,3,3)
    return weight


if __name__ == '__main__':
    # input_tensor = torch.rand(4, 1, 1, 3, 3)  # [num_experts, Cout, Cin, k, k]

    input_tensor = torch.tensor([
        [[[
            [-1., 0., 1.],
            [-1., 0., 1.],
            [-1., 0., 1.],
        ]]], 
        [[[
            [-1., 0., 1.],
            [-1., 0., 1.],
            [-1., 0., 1.],
        ]]], 
        [[[
            [-1., 0., 1.],
            [-1., 0., 1.],
            [-1., 0., 1.],
        ]]], 
        [[[
            [-1., 0., 1.],
            [-1., 0., 1.],
            [-1., 0., 1.],
        ]]], 
        ])
    print(input_tensor.shape)
    kernel_theta_list = [10.0, 20.0, 30.0, 40.0]
    output = rotate_3x3_kernel_adaptive_forloop(input_tensor.clone().detach(), input_tensor.shape[0], kernel_theta_list)

    plotshow(input_tensor[0], name='img_before0')
    plotshow(output[0], name='img_after0')
    plotshow(input_tensor[1], name='img_before1')
    plotshow(output[1], name='img_after1')
    plotshow(input_tensor[2], name='img_before2')
    plotshow(output[2], name='img_after2')
    plotshow(input_tensor[3], name='img_before3')
    plotshow(output[3], name='img_after3')