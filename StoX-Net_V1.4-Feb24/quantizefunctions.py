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


def quantize_STE_round(input_tens, bits):
    magic_number_input = 2 ** bits - 1
    out = input_tens + ((input_tens * magic_number_input).round() / magic_number_input).detach() - input_tens.detach()
    out = torch.clamp(out, -1, 1)
    return out


def quantize_STE_ceil(input_tens, bits):
    out = torch.ceil(input_tens).clamp(0, 2 ** bits - 1)
    out = torch.where(input_tens == 0, 0, out)
    return out


def quantize_STE(input_tens, bits, pos_only):
    if pos_only:
        return torch.ceil(input_tens).clamp(0, 2 ** bits - 1) / (2 ** bits - 1)
    else:
        magic_number = 2 ** bits - 1
        temp = torch.round(input_tens * magic_number) / magic_number
        return input_tens - input_tens.detach() + temp.clamp(-1, 1).detach()


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
        temp = torch.ceil(tensor).clamp(qn, qp)
        for i in range(int(bits / slice_precision)):
            temp_tensor = temp / (2 ** (slice_precision * i))
            # print((slice_precision * i), temp_tensor)
            temp_tensor = torch.floor(temp_tensor).fmod(2 ** slice_precision)
            # print(2 ** slice_precision, temp_tensor)
            bit_stream.append(temp_tensor)
        bit_stream = torch.cat(bit_stream, -1)
        bit_stream = bit_stream.fmod(2 ** slice_precision)
    else:
        magic_number_input = 2 ** bits - 1
        temp = torch.round(tensor * magic_number_input).clamp(-qp, qp)
        # temp = tensor + ((tensor * magic_number_input).round() / magic_number_input).detach() - tensor.detach()
        # temp = torch.clamp(temp, -1, 1)
        # temp = temp * magic_number_input
        for i in range(int(bits / slice_precision)):
            temp1 = temp / (2 ** (slice_precision * i))
            temp2 = torch.floor(temp1).fmod(2 ** slice_precision)
            bit_stream.append(temp-temp.detach()+temp2.detach())
        bit_stream = torch.cat(bit_stream, -1)

    return bit_stream
