from torch.autograd import Function
import torch
from functionsbase import *
from debug import tensor_stats

# 10/7/2023 Changelog
'''
Trainer py:
1. Added 3 arguments to argparse: max_time_steps, max_wb, max_ab
2. Replaced LOG_UP function with IR-Net paper
3. Added in scheduler for Time Steps, A_Bits, W_Bits

Resnet py:
1. Added two importance factors for Conv1 and Conv2 respectively
    a. These are clamped between 0, 4 and are applied as scalars
2. Activations are changed to all hardtanh

Stochastic Lib Quan py:
1. iterations, a_bits, and w_bits are now torch.tensor(1) be default, to be changed by trainer.py
2. MTJ instance now properly normalize along channels


Quantize Functions py:
1. Bits for Inputs and Weights are now passed as arguments to (Function)s 
2. Opted for torch.round instead of separate ceil and floor functions

'''
# class InputQuantize(Function):
#     @staticmethod
#     def forward(ctx, input_tens, k, t, bits):
#         ctx.save_for_backward(input_tens, k, t)
#         magic_number_input = 2 ** (bits) - 1
#         out = input_tens * magic_number_input
#         out = torch.where(out < 0, torch.floor(out), out)
#         out = torch.where(out > 0, torch.ceil(out), out)
#         out /= magic_number_input
#         out = torch.clamp(out, 0, 1)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None, None, None
#
#
# class WeightQuantize(Function):
#     @staticmethod
#     def forward(ctx, input_tens, k, t, bits):
#         ctx.save_for_backward(input_tens, k, t)
#         magic_number_weights = 2 ** (bits) - 1
#         out = input_tens * magic_number_weights
#         out = torch.where(out < 0, torch.floor(out), out)
#         out = torch.where(out > 0, torch.ceil(out), out)
#         out /= magic_number_weights
#         out = torch.clamp(out, 0, 1)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input_tens, k, t = ctx.saved_tensors
#         grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
#         return grad_input, None, None, None


class MTJBinarizeStoX(Function):
    @staticmethod
    def forward(ctx, input_tens):
        ctx.save_for_backward(input_tens)
        input_tens_tanh = torch.tanh(4 * input_tens)
        rand_tens = (2 * torch.rand_like(input_tens).cuda()) - 1
        mask1 = input_tens_tanh > rand_tens
        mask3 = input_tens == 0
        out = 1 * mask1.type(torch.float32) + -1 * (1-mask1.type(torch.float32))
        out = 0 * mask3.type(torch.float32) + out * (1-mask3.type(torch.float32))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# def streamify(x, bits):
#     negative_scalar = 2 ** (bits - 1)
#     positive_scalar = negative_scalar - 1
#
#     mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, int)
#     x = x.unsqueeze(-1)
#
#     pos_x = torch.where(x > 0, (x * positive_scalar), 0).round().int()
#     neg_x = torch.where(x < 0, (-x * negative_scalar), 0).round().int()
#
#     return pos_x.bitwise_and(mask).float().squeeze() + (-1 * neg_x.bitwise_and(mask).float()).squeeze()
#
#
# def get_scalar(dim, bits):
#     scalar_number = bits - 1
#     scalar = 2 ** torch.arange(scalar_number, -1, -1) / (2 ** scalar_number)
#     scalar = torch.stack([scalar] * int(dim / bits)).reshape(-1)
#     return scalar


def weight_quantize(weight_tensor, start_val, bits):
    magic_number = 2 ** bits - 1
    out = (2 * ((weight_tensor - torch.min(weight_tensor)) / (torch.max(weight_tensor) - torch.min(weight_tensor)))) - 1  # Bring tensor within -1, 1
    out *= magic_number
    out = floor_and_ceil(out, start_val) / magic_number
    out = out.clamp(-magic_number, magic_number)
    return out


def input_quantize(input_tensor, start_val, bits):
    magic_number = 2 ** bits - 1
    out = ((input_tensor - torch.min(input_tensor)) / (torch.max(input_tensor) - torch.min(input_tensor)))  # Bring tensor within 0, 1
    out *= magic_number
    out = floor_and_ceil(out, start_val) / magic_number
    out = out.clamp(0, magic_number)
    return out


def streamify_weights(quan_weight_tensor, bits):
    magic_number = 2 ** bits - 1
    positive_tensor = (torch.where(quan_weight_tensor > 0, quan_weight_tensor, 0) * magic_number).round().int().cuda()
    negative_tensor = (torch.where(quan_weight_tensor < 0, quan_weight_tensor.abs(), 0) * magic_number).round().int().cuda()
    positive_arange = 2 ** torch.arange(bits - 1, -1, -1).int().cuda()
    negative_arange = 2 ** torch.arange(bits - 1, -1, -1).int().cuda()
    output = torch.sign(positive_tensor.bitwise_and(positive_arange).float() + (-1 * negative_tensor.bitwise_and(negative_arange).float()))
    # output = positive_tensor.bitwise_and(positive_arange).float() + (-1 * negative_tensor.bitwise_and(negative_arange).float())
    return output


def streamify_input(input_tensor, bits):
    magic_number = 2 ** bits - 1
    positive_tensor = (torch.where(input_tensor > 0, input_tensor, 0) * magic_number).round().int().cuda()
    positive_arange = 2 ** torch.arange(bits - 1, -1, -1).int().cuda()
    output = torch.sign(positive_tensor.bitwise_and(positive_arange).float())
    # output = positive_tensor.bitwise_and(positive_arange).float()
    return output


def get_aranged_two_layers(wbs, abs):
    wbs_aranged = 2 ** torch.arange(wbs - 1, -1, -1, device='cuda:0').unsqueeze(-1).expand(wbs, abs).flatten() / (2 ** wbs - 1)
    abs_aranged = 2 ** torch.arange(abs - 1, -1, -1, device='cuda:0').repeat(wbs) / (2 ** abs - 1)
    output = wbs_aranged * abs_aranged
    return output


if __name__ == '__main__':
    print(get_aranged_two_layers(2, 2))
