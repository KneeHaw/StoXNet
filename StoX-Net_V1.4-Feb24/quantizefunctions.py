from torch.autograd import Function
import torch
from functionsbase import *
from debug import tensor_stats
import time


def quantize_STE_floor_ceil(input_tens, bits):
    magic_number_input = 2 ** bits - 1
    with torch.no_grad():
        temp1 = input_tens * magic_number_input
        temp = torch.where(temp1 > 0, torch.ceil(temp1), temp1)
        temp = torch.where(temp1 < 0, torch.floor(temp), temp)
        temp = torch.where(temp1 == 0, 0, temp)
        temp = temp / magic_number_input
    out = input_tens + temp.detach() - input_tens.detach()
    out = torch.clamp(out, -1, 1)
    return out


def quantize_STE_ceil(input_tens, bits):
    ceil = torch.ceil(input_tens).clamp(0, 2 ** bits - 1)
    out = torch.where(input_tens == 0, 0, ceil) / (2 ** bits - 1)
    return out


def create_input_stream(tensor, bits):  # Takes -1, 1 input and binarizes it accordingly. Should be gradient safe?
    magic_number = 2 ** bits - 1
    tensor = torch.ceil(tensor).clamp(0, 2 ** bits - 1)
    arranged = (1 / (2 ** torch.arange(bits, device='cuda')).unsqueeze(0))
    bit_stream = torch.floor(torch.matmul(tensor.clamp(0, magic_number).unsqueeze(-1), arranged).fmod(2))
    bit_stream = bit_stream.reshape(bit_stream.size(0), bit_stream.size(1), -1)
    return bit_stream


def input_stream(tensor, bits, slice_precision, qn, qp, pos_only):  # LSB to MSB stack dim, takes in int
    bit_stream = []
    if pos_only:
        temp_tensor = torch.ceil(tensor).clamp(qn, qp)
        for i in range(int(bits / slice_precision)):
            temp_tensor = temp_tensor / (2 ** (slice_precision * i))
            temp_tensor = torch.floor(temp_tensor)
            bit_stream.append(temp_tensor)
        bit_stream = torch.cat(bit_stream, -1)
        bit_stream = bit_stream.fmod(2 ** slice_precision)
    else:
        tensor = tensor.clamp(-qp, qp)
        temp = torch.where(tensor > 0, tensor.ceil(), tensor)
        temp = torch.where(tensor < 0, temp.floor(), temp)
        temp = torch.where(tensor == 0, 0, temp)
        negative_signs = torch.where(tensor < 0, -1, 1)
        temp = temp.abs()
        for i in range(int(bits / slice_precision)):
            temp = temp / (2 ** (slice_precision * i))
            temp = negative_signs * torch.floor(temp).fmod(2 ** slice_precision)
            bit_stream.append(temp)
        bit_stream = torch.cat(bit_stream, -1)
    return bit_stream
