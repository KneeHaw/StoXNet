import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizefunctions import *
from debug import tensor_stats, update_pt, update_txt

subarray_dimension = 256


class StoX_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
        super(StoX_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.a_bits = torch.tensor(1)
        self.w_bits = torch.tensor(1)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.iterations = torch.tensor(1)

    def conv_ADC_Less_v3_quan(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        kernel_width = filter_weights.size(dim=-1)
        kernel_height = filter_weights.size(dim=-2)
        flattened_weights = torch.flatten(filter_weights, 1)
        kernels = F.unfold(image_map, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding,
                           dilation=dilation)
        out_pixels = int(kernels.size(dim=-1) ** 0.5)
        num_chunks = (kernels.size(1) / subarray_dimension).__ceil__()
        kernel_list = torch.chunk(kernels, chunks=num_chunks, dim=1)
        weight_list = torch.chunk(flattened_weights, chunks=num_chunks, dim=1)

        non_streamed_output = 0
        for i in range(len(kernel_list)):
            working_weight = weight_list[i]
            working_kernel = kernel_list[i].transpose(-2, -1)
            for j in range(int(self.iterations)):
                linear_result = F.linear(working_kernel, working_weight).transpose(1, 2)
                non_streamed_output += self.mtj_instance(linear_result) / num_chunks


        sliced_weights = streamify_weights(flattened_weights.unsqueeze(-1), self.w_bits).split(1, -1)
        sliced_weights = torch.cat(sliced_weights, dim=0).squeeze()  # now has slices versions of each output channel, each weighted with base 2 to be later added
        sliced_weight_list = torch.chunk(sliced_weights, chunks=num_chunks, dim=1)
        # sliced_weight_list = torch.chunk(sliced_weights, chunks=(sliced_weights.size(0) / subarray_dimension).__ceil__(), dim=0)
        streamed_input = streamify_input(kernels.unsqueeze(-1), self.a_bits).split(1, -1)
        streamed_input = torch.cat(streamed_input, dim=2).squeeze()  # now has expanded to slices versions of each pixel
        streamed_kernels_list = torch.chunk(streamed_input, chunks=num_chunks, dim=1)
        # streamed_kernels_list = torch.chunk(streamed_input, chunks=(streamed_input.size(2) / subarray_dimension).__ceil__(), dim=2)

        streamed_out = 0
        for i in range(len(streamed_kernels_list)):
            working_weight_slice = sliced_weight_list[i]  # Size = (out_channs * w_slices, kernel_h * kernel_w * in_channs)
            working_kernel_stream = streamed_kernels_list[i] # Size = ((batch_size, kernel_h * kernel_w * in_channs, image_h * image_w * input_stream_length)
            # Lower dimensions are the higher values in base 2
            for j in range(int(self.iterations)):
                linear_result = F.linear(working_kernel_stream.transpose(-2, -1), working_weight_slice).transpose(-1, -2)  # output = (batch, channels * weight_slices, pixels * input_stream
                # mtj_result = self.mtj_instance(linear_result).unsqueeze(0)
                split_linear = linear_result.unsqueeze(0).split(filter_weights.size(0), -2)
                res_list = []
                for split in split_linear:
                    res_list.extend(split.split(kernels.size(-1), -1))
                temp = self.mtj_instance((torch.cat(res_list, dim=0)))
                exit()
                temp *= get_aranged_two_layers(self.w_bits, self.a_bits).view(-1, 1, 1, 1)
                streamed_out += temp.sum(dim=0) / num_chunks

        # print(tensor_stats(output), tensor_stats(new_out))
        print(tensor_stats(streamed_out), tensor_stats(non_streamed_output))
        print(torch.sum(streamed_out - non_streamed_output)/torch.numel(streamed_out))
        # exit()
        output = streamed_out.detach() + non_streamed_output - non_streamed_output.detach()

        result = F.fold(output, (out_pixels, out_pixels), (1, 1))
        return result

    def mtj_instance(self, tensor):  # Size = (Batch_sz, chann, pixels**2)
        # return tensor
        variance = torch.var(tensor, dim=-2).unsqueeze(dim=1)
        mean = torch.mean(tensor, dim=-2).unsqueeze(dim=1)
        print(tensor.size(), mean.size())
        output = (tensor - mean) / torch.pow((variance + 0.001), 0.5)
        print(tensor_stats(output))
        output = MTJBinarizeStoX().apply(output)
        return output

    def forward(self, inputs):
        w = self.weight
        a = inputs
        # print("Raw input Stats: " + tensor_stats(a))
        # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        # sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
        #                (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
        #     bw.size(0), 1, 1, 1).detach().cuda()

        # qw = WeightQuantize().apply(bw, self.k, self.t, self.w_bits).cuda()
        # qw = qw * sw
#         qa = InputQuantize().apply(a, self.k, self.t, self.a_bits).cuda()
        qw = weight_quantize(w, 0, self.w_bits)
        qa = input_quantize(a, 0, self.w_bits)
        output = self.conv_ADC_Less_v3_quan(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups).cuda()
        return output

