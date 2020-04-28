#!/usr/bin/env python3
#
# File   : cond_conv.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 12/25/2019
#
# Author of CondRotConv: Yifan Pu
# Email: utmov1776@buaa.edu.cn
# Data: 2020/04/27
#
# Distributed under terms of the MIT license.

import torch

from apex import amp
from torch import nn
from .adaptive_kernel_rotation import (
    rotate_3x3_kernel_adaptive_matrixcompute, 
    rotate_3x3_kernel_adaptive_forloop,
)

class CondConv2d(nn.Module):

    def __init__(self, num_experts, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode
        assert not bias

        self.weight = nn.Parameter(
            torch.Tensor(
                num_experts * out_channels,
                in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        )
        self.fc = nn.Linear(in_channels, num_experts)
        self.fc.zero_init = True

    @amp.float_function
    def dynaic_inference(self, x, weight):
        # TODO(Hang Gao @ 12/26): make sure passing weight to amp is necessary.
        n = x.shape[0]  # n = batch_size

        avg_x = x.mean((2, 3))  # avg_x.shape = [batch_size, Cin]
        gate_x = torch.sigmoid(self.fc(avg_x))  # gate_x.shape = [batch_size, num_experts]

        weight = torch.mm(
            gate_x,                                     # shape = [batch_size, num_experts]
            self.weight.reshape(self.num_experts, -1)   # shape = [num_experts, Cout x (Cin // groups) x k x k]
        ).reshape(                                      # result = [batch_size, Cout x (Cin // groups) x k x k]
            n * self.out_channels,                      
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        return weight  # shape = [batch_size x Cout, Cin // groups, k, k]

    def forward(self, x):
        n, _, h, w = x.shape  # n = batch_size
        weight = self.dynaic_inference(x, self.weight)  # shape = [batch_size x Cout, Cin // groups, k, k]

        out = nn.functional.conv2d(
            x.reshape(1, n * self.in_channels, h, w),   # shape = [1, batch_size x Cin, h, w]
            weight,                                     # shape = [batch_size x Cout, Cin // groups, k, k]
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n*self.groups,
            # padding_mode=self.padding_mode,  # torch.nn.functional.conv2d does not have a padding_mode argument
        )
        out = out.reshape(n, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{num_experts}, {in_channels}, {out_channels}'
             ', kernel_size={kernel_size}, stride={stride}'
             ', scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)


class CondRotConv2d(nn.Module):

    def __init__(self, num_experts, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', proportion=1):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode
        self.proportion = proportion
        assert not bias

        self.weight = nn.Parameter(
            torch.Tensor(
                num_experts * out_channels,
                in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        )
        self.fc_a = nn.Linear(in_channels, num_experts)
        self.fc_a.zero_init = True

        self.fc_theta = nn.Linear(in_channels, num_experts)
        self.fc_theta.zero_init = True

    def _combine(self, weight, gate_x_slice, num_experts):
        # [1, n] x [n, Cout x Cin // groups x k x k]
        # ---> [1, Cout x Cin // groups x k x k]
        return torch.mm(gate_x_slice.unsqueeze(0), 
                        weight.contiguous().view(num_experts, -1))


    @amp.float_function
    def dynaic_inference(self, x, weight):
        # TODO(Hang Gao @ 12/26): make sure passing weight to amp is necessary.
        batch_size = x.shape[0]  # b = batch_size

        avg_x = x.mean((2, 3))  # avg_x.shape = [batch_size, Cin]
        gate_x = torch.sigmoid(self.fc_a(avg_x))  # gate_x.shape = [batch_size, num_experts]
        theta_x = torch.sigmoid(self.fc_theta(avg_x)) * self.proportion  # theta_x.shape = [batch_size, num_experts]

        # weight.shape = [num_experts * out_channels, in_channels // self.groups, kernel_size, kernel_size]
        weight = weight.view(self.num_experts, self.out_channels, self.in_channels // self.groups, 
                             self.kernel_size, self.kernel_size)
        weight_out = torch.zeros(batch_size, self.out_channels * (self.in_channels // self.groups) \
                                 * self.kernel_size  * self.kernel_size, device=weight.device)  # initialize a empty tensor
        for idx in range(batch_size):
            weight_out[idx] = self._combine(rotate_3x3_kernel_adaptive_matrixcompute(weight, self.num_experts, 
                                       theta_x[idx]), gate_x[idx], self.num_experts)
        weight_out = weight_out.reshape(                                      
            batch_size * self.out_channels,                      
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )

        return weight_out

    def forward(self, x):
        n, _, h, w = x.shape  # n = batch_size
        weight = self.dynaic_inference(x, self.weight)  # shape = [batch_size x Cout, Cin // groups, k, k]

        out = nn.functional.conv2d(
            x.reshape(1, n * self.in_channels, h, w),   # shape = [1, batch_size x Cin, h, w]
            weight,                                     # shape = [batch_size x Cout, Cin // groups, k, k]
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n*self.groups,
            # padding_mode=self.padding_mode,  # torch.nn.functional.conv2d does not have a padding_mode argument
        )
        out = out.reshape(n, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{num_experts}, {in_channels}, {out_channels}'
             ', kernel_size={kernel_size}, stride={stride}'
             ', scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)


class CondRotConv2d_bmm(nn.Module):

    def __init__(self, num_experts, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', proportion=1):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode
        self.proportion = proportion
        assert not bias

        self.weight = nn.Parameter(
            torch.Tensor(
                num_experts * out_channels,
                in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        )
        self.fc_a = nn.Linear(in_channels, num_experts)
        self.fc_a.zero_init = True

        self.fc_theta = nn.Linear(in_channels, num_experts)
        self.fc_theta.zero_init = True

    def _combine(self, weight, gate_x_slice, num_experts):
        # [1, n] x [n, Cout x Cin // groups x k x k]
        # ---> [1, Cout x Cin // groups x k x k]
        return torch.mm(gate_x_slice.unsqueeze(0), 
                        weight.contiguous().view(num_experts, -1))

    @amp.float_function
    def dynaic_inference(self, x, weight):
        # TODO(Hang Gao @ 12/26): make sure passing weight to amp is necessary.
        batch_size = x.shape[0]  # b = batch_size

        avg_x = x.mean((2, 3))  # avg_x.shape = [batch_size, Cin]
        gate_x = torch.sigmoid(self.fc_a(avg_x))  # gate_x.shape = [batch_size, num_experts]
        theta_x = torch.sigmoid(self.fc_theta(avg_x)) * self.proportion  # theta_x.shape = [batch_size, num_experts]

        # weight.shape = [num_experts * out_channels, in_channels // self.groups, kernel_size, kernel_size]
        weight = weight.view(self.num_experts, self.out_channels, self.in_channels // self.groups, 
                             self.kernel_size, self.kernel_size)
        weight_out = torch.zeros(batch_size, self.num_experts, self.out_channels, (self.in_channels // self.groups),
                                 self.kernel_size, self.kernel_size, device=weight.device)  # initialize a empty tensor
        for idx in range(batch_size):
            weight_out[idx] = rotate_3x3_kernel_adaptive_matrixcompute(weight, self.num_experts, theta_x[idx]).unsqueeze(0)

        weight_out = torch.bmm(
            gate_x.unsqueeze(1),                      # shape = [batch_size, 1, num_experts]
            weight_out.reshape(batch_size, self.num_experts, -1)  # shape = [batch_size, num_experts, Cout x (Cin // groups) x k x k]
        ).squeeze(1).reshape(                         # result = [batch_size, 1, Cout x (Cin // groups) x k x k]
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )

        return weight_out

    def forward(self, x):
        n, _, h, w = x.shape  # n = batch_size
        weight = self.dynaic_inference(x, self.weight)  # shape = [batch_size x Cout, Cin // groups, k, k]

        out = nn.functional.conv2d(
            x.reshape(1, n * self.in_channels, h, w),   # shape = [1, batch_size x Cin, h, w]
            weight,                                     # shape = [batch_size x Cout, Cin // groups, k, k]
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n*self.groups,
            # padding_mode=self.padding_mode,  # torch.nn.functional.conv2d does not have a padding_mode argument
        )
        out = out.reshape(n, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{num_experts}, {in_channels}, {out_channels}'
             ', kernel_size={kernel_size}, stride={stride}'
             ', scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)


class CondRotConv2d_forloop(nn.Module):

    def __init__(self, num_experts, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', proportion=1):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode
        self.proportion = proportion
        assert not bias

        self.weight = nn.Parameter(
            torch.Tensor(
                num_experts * out_channels,
                in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        )
        self.fc_a = nn.Linear(in_channels, num_experts)
        self.fc_a.zero_init = True

        self.fc_theta = nn.Linear(in_channels, num_experts)
        self.fc_theta.zero_init = True

    def _combine(self, weight, gate_x_slice, num_experts):
        # [1, n] x [n, Cout x Cin // groups x k x k]
        # ---> [1, Cout x Cin // groups x k x k]
        return torch.mm(gate_x_slice.unsqueeze(0), 
                        weight.contiguous().view(num_experts, -1))


    @amp.float_function
    def dynaic_inference(self, x, weight):
        # TODO(Hang Gao @ 12/26): make sure passing weight to amp is necessary.
        batch_size = x.shape[0]  # b = batch_size

        avg_x = x.mean((2, 3))  # avg_x.shape = [batch_size, Cin]
        gate_x = torch.sigmoid(self.fc_a(avg_x))  # gate_x.shape = [batch_size, num_experts]
        theta_x = torch.sigmoid(self.fc_theta(avg_x)) * self.proportion  # theta_x.shape = [batch_size, num_experts]

        # weight.shape = [num_experts * out_channels, in_channels // self.groups, kernel_size, kernel_size]
        weight = weight.view(self.num_experts, self.out_channels, self.in_channels // self.groups, 
                             self.kernel_size, self.kernel_size)
        weight_out = torch.zeros(batch_size, self.out_channels * (self.in_channels // self.groups) \
                                 * self.kernel_size  * self.kernel_size, device=weight.device)  # initialize a empty tensor
        for idx in range(batch_size):
            weight_out[idx] = self._combine(rotate_3x3_kernel_adaptive_forloop(weight, self.num_experts, 
                                       theta_x[idx]), gate_x[idx], self.num_experts)
        weight_out = weight_out.reshape(                                      
            batch_size * self.out_channels,                      
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )

        return weight_out

    def forward(self, x):
        n, _, h, w = x.shape  # n = batch_size
        weight = self.dynaic_inference(x, self.weight)  # shape = [batch_size x Cout, Cin // groups, k, k]

        out = nn.functional.conv2d(
            x.reshape(1, n * self.in_channels, h, w),   # shape = [1, batch_size x Cin, h, w]
            weight,                                     # shape = [batch_size x Cout, Cin // groups, k, k]
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n*self.groups,
            # padding_mode=self.padding_mode,  # torch.nn.functional.conv2d does not have a padding_mode argument
        )
        out = out.reshape(n, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{num_experts}, {in_channels}, {out_channels}'
             ', kernel_size={kernel_size}, stride={stride}'
             ', scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)


class CondRotConv2d_bmm_forloop(nn.Module):

    def __init__(self, num_experts, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', proportion=1):
        super().__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.padding_mode = padding_mode
        self.proportion = proportion
        assert not bias

        self.weight = nn.Parameter(
            torch.Tensor(
                num_experts * out_channels,
                in_channels // self.groups,
                kernel_size,
                kernel_size,
            )
        )
        self.fc_a = nn.Linear(in_channels, num_experts)
        self.fc_a.zero_init = True

        self.fc_theta = nn.Linear(in_channels, num_experts)
        self.fc_theta.zero_init = True

    def _combine(self, weight, gate_x_slice, num_experts):
        # [1, n] x [n, Cout x Cin // groups x k x k]
        # ---> [1, Cout x Cin // groups x k x k]
        return torch.mm(gate_x_slice.unsqueeze(0), 
                        weight.contiguous().view(num_experts, -1))

    @amp.float_function
    def dynaic_inference(self, x, weight):
        # TODO(Hang Gao @ 12/26): make sure passing weight to amp is necessary.
        batch_size = x.shape[0]  # b = batch_size

        avg_x = x.mean((2, 3))  # avg_x.shape = [batch_size, Cin]
        gate_x = torch.sigmoid(self.fc_a(avg_x))  # gate_x.shape = [batch_size, num_experts]
        theta_x = torch.sigmoid(self.fc_theta(avg_x)) * self.proportion  # theta_x.shape = [batch_size, num_experts]

        # weight.shape = [num_experts * out_channels, in_channels // self.groups, kernel_size, kernel_size]
        weight = weight.view(self.num_experts, self.out_channels, self.in_channels // self.groups, 
                             self.kernel_size, self.kernel_size)
        weight_out = torch.zeros(batch_size, self.num_experts, self.out_channels, (self.in_channels // self.groups),
                                 self.kernel_size, self.kernel_size, device=weight.device)  # initialize a empty tensor
        for idx in range(batch_size):
            weight_out[idx] = rotate_3x3_kernel_adaptive_forloop(weight, self.num_experts, theta_x[idx]).unsqueeze(0)

        weight_out = torch.bmm(
            gate_x.unsqueeze(1),                      # shape = [batch_size, 1, num_experts]
            weight_out.reshape(batch_size, self.num_experts, -1)  # shape = [batch_size, num_experts, Cout x (Cin // groups) x k x k]
        ).squeeze(1).reshape(                         # result = [batch_size, 1, Cout x (Cin // groups) x k x k]
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )

        return weight_out

    def forward(self, x):
        n, _, h, w = x.shape  # n = batch_size
        weight = self.dynaic_inference(x, self.weight)  # shape = [batch_size x Cout, Cin // groups, k, k]

        out = nn.functional.conv2d(
            x.reshape(1, n * self.in_channels, h, w),   # shape = [1, batch_size x Cin, h, w]
            weight,                                     # shape = [batch_size x Cout, Cin // groups, k, k]
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n*self.groups,
            # padding_mode=self.padding_mode,  # torch.nn.functional.conv2d does not have a padding_mode argument
        )
        out = out.reshape(n, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{num_experts}, {in_channels}, {out_channels}'
             ', kernel_size={kernel_size}, stride={stride}'
             ', scale={scale}, zero_point={zero_point}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)