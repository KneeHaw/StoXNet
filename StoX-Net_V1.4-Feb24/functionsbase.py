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


def floor_and_ceil(tensor, start_val):
    tensor = floor().apply(tensor, start_val)
    tensor = ceil().apply(tensor, start_val)
    return tensor


class adcless_floor(Function):
    @staticmethod
    def forward(ctx, input_tens):
        return torch.floor(input_tens)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class adcless_remainder(Function):
    @staticmethod
    def forward(ctx, input_tens, number):
        return torch.remainder(input_tens, number)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class adcless_scalegrad(Function):
    @staticmethod
    def forward(ctx, input_tens, number):
        return input_tens

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / 4, None