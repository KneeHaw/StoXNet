from torch.autograd import Function
import torch
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


class WeightQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, bits):
        ctx.save_for_backward(input_tens, k, t)
        magic_number_weights = torch.tensor(2 ** (bits-1), device='cuda')
        out = input_tens * magic_number_weights
        if bits > 1:
            out = out.round()
        else:
            out = torch.where(out < 0, torch.floor(out), out)
            out = torch.where(out > 0, torch.ceil(out), out)
        out = torch.clamp(out, -magic_number_weights, magic_number_weights) / magic_number_weights
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_input, None, None, None


def create_input_stream(tensor, bits):  # Takes -1, 1 input and binarizes it accordingly. Should be gradient safe?
    magic_number = 2 ** bits - 1
    tensor = torch.round(tensor)
    arranged = (1 / (2 ** torch.arange(bits, device='cuda')).unsqueeze(0))
    bit_stream = torch.floor(torch.matmul(tensor.clamp(0, magic_number).unsqueeze(-1), arranged).fmod(2))
    bit_stream = bit_stream.reshape(bit_stream.size(0), bit_stream.size(1), -1)
    return bit_stream


class adcless_floor_remainder(Function):
    @staticmethod
    def forward(ctx, input_tens, number):
        return torch.remainder(torch.floor(input_tens), number)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def create_input_stream_adcless(tensor, bits, slice_precision, qn, qp):  # LSB to MSB stack dim, takes in int
    bit_stream = []
    tensor = tensor.clamp(qn, qp)
    for i in range(int(bits / slice_precision)):
        temp_tensor = tensor / (2 ** slice_precision) ** i
        temp_tensor = adcless_floor_remainder().apply(temp_tensor, 2 ** slice_precision)
        bit_stream.append(temp_tensor)
    bit_stream = torch.cat(bit_stream, -1)
    return bit_stream


def create_weight_sliced_adcless(tensor, bits, slice_precision, qn, qp):  # LSB to MSB stack dim, takes in int
    bit_stream = []
    tensor = tensor.clamp(qn, qp)
    tensor_negatives = torch.where(tensor < 0, -1, 1)
    tensor = tensor.abs()
    for i in range(int(bits / slice_precision)):
        temp_tensor = tensor / (2 ** slice_precision) ** i
        temp_tensor = adcless_floor_remainder().apply(temp_tensor, 2 ** slice_precision)
        bit_stream.append(tensor_negatives * temp_tensor)
    bit_stream = torch.cat(bit_stream, 0)
    return bit_stream


def gen_weight_vector_and_sum(tensor, bits):
    vector = (2 ** torch.arange(0, bits, 1, device='cuda'))
    tensor_sum = torch.stack(torch.tensor_split(tensor, bits, dim=1), dim=-1) * vector
    tensor_sum = tensor_sum.sum(-1)
    return tensor_sum


def gen_image_vector_and_sum(tensor, bits):
    vector = (2 ** torch.arange(bits, device='cuda'))
    tensor_sum = torch.stack(torch.split(tensor, bits, dim=-1)) * vector
    tensor_sum = tensor_sum.permute(1, 2, 0, 3).sum(-1)
    return tensor_sum


if __name__ == '__main__':
    pass
