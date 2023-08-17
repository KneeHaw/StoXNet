import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizefunctions import WeightQuantize, InputQuantize, MTJBinarizeStoX, MTJBinarizeADCLess

subarray_dimension = 256
adc_scalar_status = False
adc_architecture = True
adc_MTJ = False

class StoX_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None,
                 a_bit=1, w_bit=1):
        super(StoX_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                          groups, bias)
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.alpha = torch.tensor([10]).float().cuda()
        self.beta = torch.tensor([0.1]).float().cuda()
        self.ADCLess_scalar = torch.randn(1, requires_grad=True).cuda()

    def conv_ADC_Less_v3_quan(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        kernel_width = filter_weights.size(dim=-1)
        kernel_height = filter_weights.size(dim=-2)
        flattened_weights = torch.flatten(filter_weights, 1)
        kernels = F.unfold(image_map, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding,
                           dilation=dilation)

        num_chunks = (kernels.size(1) / subarray_dimension).__ceil__()
        kernel_list = torch.chunk(kernels, chunks=num_chunks, dim=1)
        weight_list = torch.chunk(flattened_weights, chunks=num_chunks, dim=1)

        output = 0
        for i in range(len(kernel_list)):
            working_weight = weight_list[i]
            working_kernel = kernel_list[i].transpose(-2, -1)
            if adc_scalar_status:
                output += self.ADCLess_scalar * (self.mtj_instance(F.linear(working_kernel, working_weight).transpose(1, 2)) / num_chunks)
            else:
                output += self.mtj_instance(F.linear(working_kernel, working_weight).transpose(1, 2)) / num_chunks

        out_pixels = int(kernels.size(dim=-1) ** 0.5)
        result = F.fold(output, (out_pixels, out_pixels), (1, 1))
        return result

    def conv_samba_v1_quan(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        kernel_width = filter_weights.size(dim=-1)
        kernel_height = filter_weights.size(dim=-2)

        flattened_weights = torch.flatten(filter_weights, 1).transpose(1, 0)
        unfolded = F.unfold(image_map, kernel_size=(kernel_height, kernel_width), padding=padding, stride=stride,
                            dilation=dilation)

        weights = self.split_tensor_samba(flattened_weights, kernel_height, kernel_width, version='weights')
        kernels = self.split_tensor_samba(unfolded, kernel_height, kernel_width, version='kernels')

        num_chunks = ((kernels.size(-2)) / subarray_dimension).__ceil__()
        weight_list = torch.chunk(weights, chunks=num_chunks, dim=-2)
        kernel_list = torch.chunk(kernels, chunks=num_chunks, dim=-2)

        output = 0
        for i in range(len(weight_list)):
            working_weight = weight_list[i].transpose(-1, -2)
            working_kernel = kernel_list[i].transpose(-1, -2)
            if adc_scalar_status:
                output += self.ADCLess_scalar * (self.mtj_instance(F.linear(working_kernel, working_weight).transpose(1, 2)) / num_chunks)
            else:
                output += self.mtj_instance(F.linear(working_kernel, working_weight).transpose(-1, -2)) / num_chunks

        out_pixels = int(output.size(-1) ** 0.5)
        output = F.fold(output, (out_pixels, out_pixels), (1, 1))
        return output

    @staticmethod
    def split_tensor_samba(tensor, kernel_height, kernel_width, version):
        kernel_size = kernel_height * kernel_width
        outer_list = []
        for i in range(kernel_height):
            inner_list = []
            for j in range(kernel_width):
                start_index = j + i * kernel_width
                if version == 'kernels':
                    temp_split = tensor[:, start_index::kernel_size, :]
                elif version == 'weights':
                    temp_split = tensor[start_index::kernel_size, :]
                inner_list.append(temp_split)
            outer_list.append(torch.concat(inner_list, dim=-2))
        out = torch.concat(outer_list, dim=-2)
        return out

    def mtj_instance(self, tensor):
        variance = torch.var(tensor, dim=-2).unsqueeze(dim=1)
        mean = torch.mean(tensor, dim=-2).unsqueeze(dim=1)
        output = (tensor - mean) / torch.pow((variance + 0.001), 0.5)
        if adc_MTJ:
            output = MTJBinarizeADCLess().apply(output, self.alpha, self.beta)
        else:
            output = MTJBinarizeStoX().apply(output, self.alpha, self.beta)
        return output

    def forward(self, inputs):
        w = self.weight
        a = inputs
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach().cuda()

        qw = WeightQuantize().apply(bw, self.k, self.t).cuda()
        qw = qw * sw
        qa = InputQuantize().apply(a, self.k, self.t).cuda()

        if adc_architecture:
            output = self.conv_ADC_Less_v3_quan(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups).cuda()
        else:
            output = self.conv_samba_v1_quan(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups).cuda()
        return output

