from torch.autograd import Function
import torch

num_bits_input = 1
magic_number_input = 2 ** num_bits_input - 1
magic_number_input2 = magic_number_input / 2

num_bits_weights = 1
magic_number_weights = 2 ** num_bits_weights - 1
magic_number_weights2 = magic_number_weights / 2

class InputQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t):
        ctx.save_for_backward(input_tens, k, t)
        out = input_tens * magic_number_input
        out = torch.where(out < magic_number_input2, torch.floor(out), out)
        out = torch.where(out > magic_number_input2, torch.ceil(out), out)
        out /= magic_number_input
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh((input_tens - .5) * t), 2)) * grad_output
        return grad_input, None, None


class WeightQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t):
        ctx.save_for_backward(input_tens, k, t)
        out = input_tens * magic_number_weights
        out = torch.where(out < magic_number_weights2, torch.floor(out), out)
        out = torch.where(out > magic_number_weights2, torch.ceil(out), out)
        out /= magic_number_weights
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh((input_tens - .5) * t), 2)) * grad_output
        return grad_input, None, None


class MTJBinarizeADCLess(Function):
    @staticmethod
    def forward(ctx, input_tens, a, b):
        ctx.save_for_backward(input_tens, a, b)
        out = torch.sign(input_tens)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, a, b = ctx.saved_tensors
        grad_input = a * b * (1 - torch.pow(torch.tanh((input_tens - .5) * b), 2)) * grad_output
        return grad_input, None, None


class MTJBinarizeStoX(Function):
    @staticmethod
    def forward(ctx, input_tens, a, b):
        ctx.save_for_backward(input_tens, a, b)
        rand_tens = torch.rand_like(input_tens, device='cuda:0')
        input_tens_tanh = (torch.tanh(8 * (input_tens - .5)) + 1) / 2
        mask1 = input_tens_tanh > rand_tens
        out = (1 * mask1.type(torch.float32)) + (0 * (1-mask1.type(torch.float32)))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, a, b = ctx.saved_tensors
        grad_input = a * b * (1 - torch.pow(torch.tanh((input_tens - .5) * b), 2)) * grad_output
        return grad_input, None, None