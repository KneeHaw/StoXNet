import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizefunctions import Quan_IR, MTJ_Instance_Streamed, increment_msb, QuanNew
from debug import tensor_stats, find_empty_chunking

class StoX_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None,
                 a_bits=4, w_bits=4, subarray_dimension=256):
        super(StoX_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                          groups, bias)
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.subarray_dimension = subarray_dimension
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.alpha = torch.tensor([10]).float().cuda()
        self.beta = torch.tensor([0.1]).float().cuda()
        self.gamma = torch.tensor([10]).float().cuda()
        self.delta = torch.tensor([0.1]).float().cuda()

    def conv_partial_sums(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        kernel_width = filter_weights.size(dim=-1)
        kernel_height = filter_weights.size(dim=-2)
        flattened_weights = torch.flatten(filter_weights, 1)  # Dimensions = [out_channels, in_channels * width * height], hardware = out_chann cols with rows of in_channel 1's, then 2's, then 3's 'pixels'
        kernels = F.unfold(image_map, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding,
                           dilation=dilation)  # Dimensions = [batch size, in_chann * kernel_height * kernel_width, num kernels = image area] middle dim = lines up kernel vectors with weights

        num_chunks = (kernels.size(1) / self.subarray_dimension).__ceil__()
        kernel_list = torch.chunk(kernels, chunks=num_chunks, dim=1)
        weight_list = torch.chunk(flattened_weights, chunks=num_chunks, dim=1)
        output = 0
        for i in range(len(kernel_list)):
            working_weight = weight_list[i]
            working_kernel = kernel_list[i].transpose(-2, -1)
            temp = F.linear(working_kernel, working_weight)
            output += self.mtj_instance(temp.transpose(1, 2)) / num_chunks

        out_pixels = int(kernels.size(dim=-1) ** 0.5)
        result = F.fold(output, (out_pixels, out_pixels), (1, 1))
        return result

    def mtj_instance(self, tensor):
        variance = torch.var(tensor, dim=-2).unsqueeze(dim=1)
        mean = torch.mean(tensor, dim=-2).unsqueeze(dim=1)
        output = (tensor - mean) / torch.pow((variance + 0.001), 0.5)
        output = MTJ_Instance_Streamed().apply(output, self.gamma, self.delta)
        return output

    def forward(self, inputs):
        w = self.weight
        raw_a = inputs
        norm_w = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        norm_w = norm_w / norm_w.view(norm_w.size(0), -1).std(-1).view(norm_w.size(0), 1, 1, 1)

        a_quan = QuanNew().apply(raw_a, self.k, self.t, self.a_bits)
        w_quan = QuanNew().apply(norm_w, self.k, self.t, self.w_bits)
        # a and w quan are in the range of (-MSB, +MSB) excluding zeros save for padding
        # for IR_Quan, these values are out of range (expects (-1, 1)

        output = 0
        for i in range(self.a_bits):
            a_msb = (2 * (self.a_bits - (i + 2))) / (2 * (self.a_bits - 1) - 1)
            a_temp = Quan_IR().apply(a_quan, self.alpha, self.beta, a_msb)

            for j in range(self.w_bits):
                w_msb = (2 * (self.w_bits - (j + 2))) / (2 * ((self.w_bits - 1) - 1))
                w_temp = Quan_IR().apply(w_quan, self.alpha, self.beta, w_msb)
                output += a_msb * w_msb * self.conv_partial_sums(a_temp, w_temp, self.bias, self.stride, self.padding, self.dilation, self.groups)
                w_quan = increment_msb(w_quan, w_temp, w_msb)

            a_quan = increment_msb(a_quan, a_temp, a_msb)

        print(self.k, self.t, self.alpha, self.beta)
        return output

