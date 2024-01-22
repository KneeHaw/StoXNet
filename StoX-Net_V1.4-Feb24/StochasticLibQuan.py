import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizefunctions import *
import random
from debug import tensor_stats, update_txt, plot_tensor_hist, plot_tensor_plot
import time

subarray_dimension = 256


def get_chunks(inc, sub):
    return (inc * 9 / sub).__ceil__()


class StoX_MTJ(nn.Module):
    def __init__(self, channels):
        super(StoX_MTJ, self).__init__()
        self.bn = nn.BatchNorm1d(channels)
        # self.scalar = nn.Parameter(torch.tensor(4., device='cuda'), requires_grad=True)

    def forward(self, input_tens):
        normed_input = self.bn(input_tens)
        out = MTJInstance().apply(normed_input, 4)
        return out


class StoX_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stox_params, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
        super(StoX_Conv2d, self).__init__()
        # StoX_Params order -> [a_bits, w_bits, a_stream_width, w_slice_width, subarray_size, time_steps]

        # The class needs all of these parameters to simulate convolutions.
        self.w_bits = stox_params[1]
        self.w_bits_per_slice = stox_params[3]
        self.w_slices = int(self.w_bits / self.w_bits_per_slice)

        self.a_bits = stox_params[0]
        self.a_bits_per_stream = stox_params[2]
        self.a_slices = int(self.a_bits / self.a_bits_per_stream)

        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.bias = bias
        self.k = torch.tensor(1)
        self.t = torch.tensor(1)
        self.kernel_size = kernel_size
        self.iterations = stox_params[5]
        self.num_chunks = get_chunks(in_channels, stox_params[4])

        # Initialize weights according to number of slices and channels.
        # Every slice will distribute base 2 weighting.
        # Given 4bit weight, 2 slices would be 2bits each with the smaller being [0, 1, 2, 3] and the larger being [0, 4, 8, 12].
        self.weight = nn.Parameter(torch.empty(out_channels * self.w_slices, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

        # Either every subarray can have its own set of normed MTJs, or all of the subarrays for a layer can have the
        # same MTJ optimized for the layer.
        self.MTJs = nn.ModuleList(StoX_MTJ(out_channels * self.w_slices) for i in range(self.num_chunks))
        self.MTJ = StoX_MTJ(out_channels * self.w_slices)

    def StoX_hardware_Conv(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        flattened_weights = torch.flatten(filter_weights, 1)
        kernel_list = torch.chunk(image_map, chunks=self.num_chunks, dim=1)
        weight_list = torch.chunk(flattened_weights, chunks=self.num_chunks, dim=1)

        output = 0
        for i, working_weight in enumerate(weight_list):
            working_kernel = kernel_list[i].transpose(-2, -1)
            for k in range(int(self.iterations)):
                # Size = [Batches, Out_Chann * w_slices, Tot_Pix * a_slices]
                linear_temp = F.linear(working_kernel, working_weight).transpose(1, 2)
                output += self.MTJ(linear_temp)

        # output = output / self.num_chunks / self.iterations
        output = output
        # Generate LSB to MSB vectors for S&A
        if self.w_slices > 1:
            output = gen_weight_vector_and_sum(output, bits=self.w_slices)
        if self.a_slices > 1:
            output = gen_image_vector_and_sum(output, self.a_bits)

        out_pixels = int((output.size(dim=-1)) ** 0.5)  # Image size to map back to
        result = F.fold(output, (out_pixels, out_pixels), (1, 1))  # Fold result into image
        return result

    def forward(self, inputs):
        # Size = [out_channels, in_channels, kernel_height, kernel_width]
        # [MSB . . . LSB]
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        qw = WeightQuantize().apply(bw, self.k, self.t, self.w_bits_per_slice)
        # qw = create_weight_sliced_adcless(w, self.w_bits, self.w_bits_per_slice, -(2 ** (self.w_bits - 1)), 2 ** (self.w_bits - 1) -1)

        # Size = [batch_size, in_channels * kern_h * kern_w, pixel_height * pixel_width]
        a = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)
        if self.a_bits == 1:
            # Keep zeros and give everything else -1 or +1
            qa = quantize_STE_ceil(a, self.a_bits)  # Round to 0 and 1
            """
            *TODO* Create small "cutoff" point where it should be rounded to zero. Currently is sign function, change 
            that! There are papers on this "cutoff" point...
            """
        else:
            # Bit stream [LSB . . . MSB]
            # Size = [batch_size, in_channels * k_h * k_w, p_h * p_w, slices]
            # qa = quantize_STE_ceil(a, self.a_bits)
            qa = create_input_stream(a, self.a_bits)  #
            # qa = create_input_stream_adcless(a, self.a_bits, self.a_bits_per_stream, 0, 2 ** self.a_bits - 1)

        output1 = self.StoX_hardware_Conv(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # output1 = F.conv2d(qa, qw, None, self.stride, self.padding, self.dilation, self.groups)
        return output1


class StoX_Linear(nn.Linear):
    # NOT IMPLEMENTED
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
