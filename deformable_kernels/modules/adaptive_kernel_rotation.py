import math
import torch
import matplotlib.pyplot as plt 


"""
We only consider Bilinear 3x3 resample scenario.

The last kernel rotation function only rotates a batch of Cout x Cin x k x k kernels with same degree.

In this function, we will rotation num_experts Cout x Cin x k x k kernels with different degrees respectively.
"""

def rotate_3x3_kernel_adaptive_forloop(weights, num_experts, kernel_theta_vector):
    """
    Args:
        weights: tensor, shape = [num_experts, Cout, Cin, k, k]
        num_experts: number of experts
        kernel_theta: a list of float number with the size of num_experts
    """
    assert(weights.shape[3] == 3)
    assert(weights.shape[4] == 3)
    assert(max(kernel_theta_vector) <= 45.0)
    assert(min(kernel_theta_vector) >= - 45.0)

    for idx in range(num_experts):

        kernel_theta = kernel_theta_vector[idx]
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


def rotate_3x3_kernel_adaptive_matrixcompute(weights, num_experts, kernel_theta):
    """
    Args:
        weights: tensor, shape = [num_experts, Cout, Cin, k, k]
        num_experts: number of experts
        kernel_theta: a tensor vector of float number with the size of num_experts
    Return:
        weights: rotated weights tensor, shape = [num_experts, Cout, Cin, 3, 3]
    """
    assert(weights.shape[3] == 3)
    assert(weights.shape[4] == 3)
    assert(max(kernel_theta) <= 45.0)
    assert(min(kernel_theta) >= - 45.0)

    is_clockwise = kernel_theta < 0
    kernel_theta = torch.abs(kernel_theta)

    x_vector = torch.cos(kernel_theta / 180. * math.pi)
    y_vector = torch.sin(kernel_theta / 180. * math.pi)

    a_vector = x_vector - y_vector
    b_vector = torch.mul(x_vector, y_vector)
    c_vector = x_vector + y_vector

    alpha = torch.zeros(num_experts * 9, num_experts * 9)
    for idx in range(num_experts):
        x, y = x_vector[idx], y_vector[idx]
        a, b, c = a_vector[idx], b_vector[idx], c_vector[idx]
        if is_clockwise[idx] == False:
            # anti-clockwise (theta > 0)
            sub_alpha = torch.tensor([[ a, 1-a, 0.,  0.,     0., 0.,  0.,  0., 0.],           # w'(-1, 1)
                                      [0., x-b,  b,  0., 1-c+b, y-b,  0.,  0., 0.],           # w'( 0, 1)
                                      [0.,  0.,  a,  0.,    0., 1-a,  0.,  0., 0.],           # w'( 1, 1)
                                      [ b, y-b, 0., x-b, 1-c+b,  0.,  0.,  0., 0.],           # w'(-1, 0)
                                      [0.,  0., 0.,  0.,    1.,  0.,  0.,  0., 0.],           # w'( 0, 0)
                                      [0.,  0., 0.,  0., 1-c+b, x-b,  0., y-b,  b],           # w'( 1, 0)
                                      [0.,  0., 0., 1-a,    0.,  0.,   a,  0., 0.],           # w'(-1,-1)
                                      [0.,  0., 0., y-b, 1-c+b,  0.,   b, x-b, 0.],           # w'( 0,-1)
                                      [0.,  0., 0.,  0.,    0.,  0.,  0., 1-a,  a]])   # w'( 1,-1).cuda()
        else:
            # clockwise (theta < 0)
            sub_alpha = torch.tensor([[ a,  0., 0., 1-a,     0., 0.,  0.,  0., 0.],           # w'(-1, 1)
                                      [ b, x-b, 0., y-b, 1-c+b,  0.,  0.,  0., 0.],           # w'( 0, 1)
                                      [0., 1-a,  a,  0.,    0.,  0.,  0.,  0., 0.],           # w'( 1, 1)
                                      [0.,  0., 0., x-b, 1-c+b,  0.,   b, y-b, 0.],           # w'(-1, 0)
                                      [0.,  0., 0.,  0.,    1.,  0.,  0.,  0., 0.],           # w'( 0, 0)
                                      [0., y-b,  b,  0., 1-c+b, x-b,  0.,  0., 0.],           # w'( 1, 0)
                                      [0.,  0., 0.,  0.,    0.,  0.,   a, 1-a, 0.],           # w'(-1,-1)
                                      [0.,  0., 0.,  0., 1-c+b, y-b,  0., x-b,  b],           # w'( 0,-1)
                                      [0.,  0., 0.,  0.,    0., 1-a,  0.,  0.,  a]])   # w'( 1,-1).cuda()
        alpha[idx * 9:(idx+1) * 9, idx * 9:(idx+1) * 9] = sub_alpha

    _, Cout, Cin, _, _ = weights.shape  # [num_experts, Cout, Cin, 3, 3]
    weights = weights.transpose(0, 1).transpose(1, 2)
    # ---> [Cout, num_experts, Cin, 3, 3]
    # ---> [Cout, Cin, num_experts, 3, 3]
    weights = weights.contiguous().view(Cout * Cin, num_experts * 9)
    # ---> [Cout * Cin, num_experts * 9]
    weights = weights.transpose(0, 1)
    # ---> [num_experts * 9, Cout * Cin]
    weights = torch.mm(alpha, weights)
    # [num_experts * 9, num_experts * 9] x [num_experts * 9, Cout * Cin] ---> [num_experts * 9, Cout * Cin]
    weights = weights.transpose(0, 1).view(Cout, Cin, num_experts, 3, 3)
    # ---> [Cout * Cin, num_experts * 9]
    # ---> [Cout, Cin, num_experts, 3, 3]
    weights = weights.transpose(1, 2).transpose(0, 1)
    # ---> [Cout, num_experts, Cin, 3, 3]
    # ---> [num_experts, Cout, Cin, 3, 3]

    return weights


def _plotshow(input, i=0, j=0, name='before'):
    plt.matshow(input[i][j].numpy())
    plt.show()
    plt.savefig(f'images/{name}.png')


if __name__ == '__main__':

    input_tensor = torch.tensor([
        [
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
        ],
        [
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
        ],
        [
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
        ],
        [
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
            [
                [
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                    [-1., 0., 1.],
                ],
                [
                    [ 1., 1., 1.],
                    [ 0., 0., 0.],
                    [-1.,-1.,-1.],
                ],
            ],
        ],
        ])
    input_tensor = torch.rand(4, 128, 128, 3, 3)  # [num_experts, Cout, Cin, k, k]
    print(input_tensor.shape)
    kernel_theta_list = [10.0, 40.0, -10.0, -40.0,]

    import time
    start = time.time()
    output = rotate_3x3_kernel_adaptive_forloop(input_tensor.clone().detach(), input_tensor.shape[0], kernel_theta_list)
    end = time.time()
    print(f'for loop time      : {end - start}')

    start = time.time()
    output = rotate_3x3_kernel_adaptive_matrixcompute(input_tensor.clone().detach(), input_tensor.shape[0], kernel_theta_list)
    end = time.time()
    print(f'matrix compute time: {end - start}')


    # _plotshow(input_tensor[0], name='img_before0')
    # _plotshow(output[0], name='img_after0')
    # _plotshow(input_tensor[1], name='img_before1')
    # _plotshow(output[1], name='img_after1')
    # _plotshow(input_tensor[2], name='img_before2')
    # _plotshow(output[2], name='img_after2')
    # _plotshow(input_tensor[3], name='img_before3')
    # _plotshow(output[3], name='img_after3')
    # _plotshow(input_tensor[4], name='img_before4')
    # _plotshow(output[4], name='img_after4')
    # _plotshow(input_tensor[5], name='img_before5')
    # _plotshow(output[5], name='img_after5')
    # _plotshow(input_tensor[6], name='img_before6')
    # _plotshow(output[6], name='img_after6')
    # _plotshow(input_tensor[7], name='img_before7')
    # _plotshow(output[7], name='img_after7')