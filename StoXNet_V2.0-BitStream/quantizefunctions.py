from abc import ABC

from torch.autograd import Function
import torch
from debug import find_empty_chunking, tensor_stats
import torch.nn.functional as F

bits = 4


class QuanNew(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, bits):
        ctx.save_for_backward(input_tens, k, t)
        scalar = 2 ** (bits - 1) - 1
        mask_min = input_tens < -scalar
        mask_negative = input_tens < 0
        mask_positive = input_tens > 0
        mask_max = input_tens > scalar

        pos_piece = (input_tens * mask_positive.type(torch.float32)) + (0 * (1 - mask_negative.type(torch.float32)))
        pos_piece = (scalar * mask_max.type(torch.float32)) + (pos_piece * (1 - mask_max.type(torch.float32)))
        pos_piece = torch.ceil(pos_piece)

        neg_piece = (input_tens * mask_negative.type(torch.float32)) + (0 * (1 - mask_negative.type(torch.float32)))
        neg_piece = (-scalar * mask_min.type(torch.float32)) + (neg_piece * (1 - mask_min.type(torch.float32)))
        neg_piece = torch.floor(neg_piece)

        quantized_piece = (neg_piece + pos_piece) / scalar
        return quantized_piece

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_input, None, None, None, None


class MTJ_Instance_Streamed(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t):
        ctx.save_for_backward(input_tens, k, t)
        rand_tens = ((2 * torch.rand_like(input_tens)) - 1)
        input_tens_tanh = (torch.tanh(4 * input_tens))
        mask1 = input_tens_tanh > rand_tens
        mask2 = input_tens == 0
        out = (1 * mask1.type(torch.float32)) + (-1 * (1 - mask1.type(torch.float32)))
        out = (0 * mask2.type(torch.float32)) + (out * (1 - mask2.type(torch.float32)))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_input, None, None


def increment_msb(input_tens, temp, msb):
    mask3 = temp == 1
    mask4 = temp == -1
    quantized_piece = ((input_tens - msb) * mask3.type(torch.float32)) + (input_tens * (1 - mask3.type(torch.float32)))
    quantized_piece = ((quantized_piece + msb) * mask4.type(torch.float32)) + (quantized_piece * (1 - mask4.type(torch.float32)))
    return quantized_piece


class Quan_IR(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, msb):
        ctx.save_for_backward(input_tens, k, t)
        mask1 = input_tens >= msb
        mask2 = input_tens <= -msb
        temp = 1 * mask1.type(torch.float32) + (-1) * (mask2.type(torch.float32)) + (0 * (1 - mask1.type(torch.float32) * (1 -mask2.type(torch.float32))))
        return temp

    @staticmethod
    def backward(ctx, grad_output):
        #input_tens, k, t = ctx.saved_tensors
        #grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_output, None, None, None, None
