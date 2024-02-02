import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizefunctions import *
from debug import tensor_stats, update_pt, update_txt

subarray_dimension = 128


def get_chunks(inc, sub):
    return (inc * 9 / sub).__ceil__()


class StoX_MTJ(nn.Module):
    def __init__(self, channels):
        super(StoX_MTJ, self).__init__()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, input_tens):
        # update_txt(tensor_stats(tensor), './raw_tens.txt')
        normed_input = self.bn(input_tens)
        # update_txt(tensor_stats(output), './norm_tens.txt')

        input_tens_tanh = torch.tanh(4 * normed_input)
        rand_tens = ((2 * torch.rand_like(normed_input, device='cuda:0')) - 1)
        mask1 = input_tens_tanh > rand_tens
        mask3 = normed_input == 0
        out = 1 * (mask1.type(torch.float32)) + -1 * (1 - mask1.type(torch.float32))
        out = 0 * mask3.type(torch.float32) + out * (1 - mask3.type(torch.float32))
        return normed_input.clamp(-1, 1) - normed_input.clamp(-1, 1).detach() + out.detach()


class StoX_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, a_bits=torch.tensor(4), w_bits=torch.tensor(4)):
        super(StoX_Conv2d, self).__init__()
        self.w_slices = 2
        self.weight = nn.Parameter(torch.empty(int(out_channels * self.w_slices), in_channels, kernel_size, kernel_size))
        self.num_chunks = get_chunks(in_channels, subarray_dimension)
        self.bns = nn.ModuleList(StoX_MTJ(int(out_channels * self.w_slices)) for i in range(self.num_chunks))
        self.bn = StoX_MTJ(int(out_channels * self.w_slices))
        nn.init.kaiming_normal_(self.weight)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.k = torch.tensor(1)
        self.t = torch.tensor(1)
        self.iterations = torch.tensor(1)

    def StoX_hardware_Conv(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        kernel_width = filter_weights.size(dim=-1)
        kernel_height = filter_weights.size(dim=-2)
        flattened_weights = torch.flatten(filter_weights, 1)
        kernels = F.unfold(image_map, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding,
                           dilation=dilation)  # Dims = (batch_sz, Kh * Kw * chans, out_pixels)
        output = 0
        # scalar = 2 ** (self.a_bits - (1 + i)) / (2 ** self.a_bits - 1)
        kernel_list = torch.chunk(kernels, chunks=self.num_chunks, dim=1)
        weight_list = torch.chunk(flattened_weights, chunks=self.num_chunks, dim=1)

        for i, working_weight in enumerate(weight_list):
            working_kernel = kernel_list[i].transpose(-2, -1)
            for k in range(int(self.iterations)):
                linear_temp = F.linear(working_kernel, working_weight).transpose(1, 2)
                pre_output = self.bn(linear_temp) / self.num_chunks / self.iterations
                output = output + gen_weight_vector_and_sum(pre_output, bits=(int(self.w_slices)))
        out_pixels = int((kernels.size(dim=-1)) ** 0.5)
        result = F.fold(output, (out_pixels, out_pixels), (1, 1))
        return result

    def forward(self, inputs):
        w = self.weight
        a = inputs

        if self.a_bits == 1:
            qa = quantize_STE_floor_ceil(a, self.a_bits)
        else:
            qa = quantize_STE_round(a, self.a_bits)

        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        qw = WeightQuantize().apply(bw, self.k, self.t, self.w_bits / self.w_slices)

        output1 = self.StoX_hardware_Conv(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # output1 = F.conv2d(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print(tensor_stats(output1), tensor_stats(qw), tensor_stats(qa))

        return output1


class StoX_Linear(nn.Linear):
    def __init__(self, in_feat, out_feat):
        super(StoX_Linear, self).__init__(in_feat, out_feat)

        self.iterations = torch.tensor(32)
        self.w_bits = torch.tensor(4)
        self.a_bits = torch.tensor(4)

    def mtj_instance(self, tensor):  # Size = (Batch_sz, 10)
        variance = torch.var(tensor, dim=0).view(1, -1)
        mean = torch.mean(tensor, dim=0).view(1, -1)
        output = (tensor - mean) / torch.pow((variance + 0.001), 0.5)
        output = MTJ_STE(output, self.a_bits, self.w_bits, False)
        return output

    def forward(self, tensor):
        w = self.weight
        a = tensor

        bw = w - w.mean(-1).view(w.size(0), 1)
        bw = bw / bw.std(-1).view(bw.size(0), 1)

        qw = quantize_STE_floor_ceil(bw, self.w_bits)
        qa = quantize_STE_round(a, self.a_bits)
        result = 0
        for i in range(int(self.iterations)):
            result = result + self.mtj_instance(F.linear(input=qa, weight=qw, bias=None)) / self.iterations

        return result
