from torch.autograd import Function
import torch

num_bits = 1
magic_number = 2 ** (num_bits - 1)


class QQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t):
        ctx.save_for_backward(input_tens, k, t)
        out = input_tens * magic_number
        out = torch.where(out < 0, torch.floor(out), out)
        out = torch.where(out > 0, torch.ceil(out), out)
        out /= magic_number
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_input, None, None


class mtj_binarize_adcless(Function):
    @staticmethod
    def forward(ctx, input_tens, a, b):
        ctx.save_for_backward(input_tens, a, b)
        out = torch.sign(input_tens)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, a, b = ctx.saved_tensors
        grad_input = a * b * (1 - torch.pow(torch.tanh(input_tens * b), 2)) * grad_output
        return grad_input, None, None


class mtj_binarize_stoX(Function):
    @staticmethod
    def forward(ctx, input_tens, a, b):
        ctx.save_for_backward(input_tens, a, b)
        rand_tens = (2 * torch.rand_like(input_tens, device='cuda:0')) - 1
        input_tens_tanh = torch.tanh(4 * input_tens)
        mask1 = input_tens_tanh > rand_tens
        mask3 = input_tens == 0
        out = 1 * mask1.type(torch.float32) + (-1) * (1-mask1.type(torch.float32))
        out = 0 * mask3.type(torch.float32) + (out) * (1-mask3.type(torch.float32))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, a, b = ctx.saved_tensors
        grad_input = a * b * (1 - torch.pow(torch.tanh(input_tens * b), 2)) * grad_output
        return grad_input, None, None