from torch.autograd import Function
import torch
import torch.nn.functional as F
from debug import tensor_stats


class ceil(Function):
    @staticmethod
    def forward(ctx, input_tens, start_val):
        out = torch.where(input_tens > start_val, torch.ceil(input_tens), input_tens)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class floor(Function):
    @staticmethod
    def forward(ctx, input_tens, start_val):
        out = torch.where(input_tens < start_val, torch.floor(input_tens), input_tens)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class clamp(Function):
    @staticmethod
    def forward(ctx, input_tens, min_val, max_val):
        out = input_tens.clamp(min_val, max_val)
        return out

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None


def floor_and_ceil(tensor, start_val):
    tensor = floor().apply(tensor, start_val)
    tensor = ceil().apply(tensor, start_val)
    return tensor