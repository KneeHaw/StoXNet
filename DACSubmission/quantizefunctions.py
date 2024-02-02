from torch.autograd import Function
import torch
from functionsbase import *
from debug import tensor_stats


class InputQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, bits):
        ctx.save_for_backward(input_tens, k, t)
        magic_number_input = 2 ** bits - 1
        out = input_tens * magic_number_input

        out = out + out.round().detach() - out.detach()

        out /= magic_number_input
        out = torch.clamp(out, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_output, None, None, None


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


class WeightQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, bits):
        ctx.save_for_backward(input_tens, k, t)
        magic_number_weights = 2 ** bits - 1
        out = input_tens * magic_number_weights

        # print(bits)
        out = torch.where(out < 0, torch.floor(out), out)
        out = torch.where(out > 0, torch.ceil(out), out)

        out /= magic_number_weights
        out = torch.clamp(out, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_input, None, None, None


def MTJ_STE(input_tens, a_bits, w_bits, scale_bool: bool):
    if scale_bool:
        exit()
        bit_num = a_bits * w_bits
        with torch.no_grad():
            input_tens_tanh = F.tanh(4 * input_tens) * bit_num
            rand_tens = ((2 * torch.rand_like(input_tens, device='cuda:0')) - 1) * bit_num
            mask1 = input_tens_tanh > rand_tens
            mask2 = input_tens_tanh.abs() <= 1
            mask3 = input_tens == 0
            out = input_tens_tanh.ceil() * (mask1.type(torch.float32) * (1 - mask2.type(torch.float32))) + input_tens_tanh.floor() * ((1 - mask1.type(torch.float32)) * (1 - mask2.type(torch.float32))) + 1 * (mask1.type(torch.float32) * mask2.type(torch.float32)) + -1 * ((1 - mask1.type(torch.float32)) * mask2.type(torch.float32))
            out = 0 * mask3.type(torch.float32) + out * (1 - mask3.type(torch.float32))
            out = out.div(bit_num)
        return F.hardtanh(input_tens) - F.hardtanh(input_tens).detach() + out.detach()
    else:
        with torch.no_grad():
            input_tens_tanh = torch.tanh(4 * input_tens)
            rand_tens = ((2 * torch.rand_like(input_tens, device='cuda:0')) - 1)
            mask1 = input_tens_tanh > rand_tens
            mask3 = input_tens == 0
            out = 1 * (mask1.type(torch.float32)) + -1 * (1-mask1.type(torch.float32))
            out = 0 * mask3.type(torch.float32) + out * (1 - mask3.type(torch.float32))
        return F.hardtanh(input_tens) - F.hardtanh(input_tens).detach() + out.detach()


def get_aranged_two_layers(wbs, abs):
    wbs_aranged = 2 ** torch.arange(abs - 1, -1, -1, device='cuda:0').unsqueeze(-1).expand(wbs, abs).flatten() / (2 ** wbs - 1)
    abs_aranged = 2 ** torch.arange(wbs - 1, -1, -1, device='cuda:0').repeat(wbs) / (2 ** abs - 1)
    return torch.mul(wbs_aranged, abs_aranged)


def get_aranged_one_layer(bs):
    bs_aranged = 2 ** torch.arange(bs - 1, -1, -1, device='cuda:0') / (2 ** bs - 1)
    return bs_aranged


def streamify(x, bits):
    scalar = 2 ** bits - 1
    scalar = scalar - 1

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, int)
    # mask = torch.tensor([0, 0 , 0, 1], device='cuda')
    x = x.unsqueeze(-1)

    pos_x = torch.where(x > 0, (x * scalar), 0).round().int()
    neg_x = torch.where(x < 0, (-x * scalar), 0).round().int()

    return (pos_x.bitwise_and(mask).float().squeeze() + (-1 * neg_x.bitwise_and(mask).float()).squeeze()).clamp(-1, 1)


def gen_weight_vector_and_sum(tensor, bits):
    scalar = 2 ** bits - 1
    vector = (2 ** torch.arange(bits - 1, -1, -1)).cuda().unsqueeze(-1).repeat(1, int(tensor.size(1)/bits)).flatten() / scalar
    tensor_split = torch.tensor_split(tensor * vector.view(1, -1, 1), sections=bits, dim=1)
    tensor_stack = torch.stack(tensor_split, dim=-1)
    tensor_sum = torch.sum(tensor_stack, dim=-1)
    return tensor_sum


def gen_image_vector_and_sum(tensor, bits):
    scalar = 2 ** bits - 1
    vector = (2 ** torch.arange(bits - 1, -1, -1)).cuda().unsqueeze(-1).repeat(1, int(tensor.size(-1) / bits)).flatten() / scalar
    tensor_split = torch.tensor_split(tensor * vector.view(1, 1, -1), sections=bits, dim=-1)
    tensor_stack = torch.stack(tensor_split, dim=-1)
    tensor_sum = torch.sum(tensor_stack, dim=-1)
    # print(vector.requires_grad, tensor_stack.requires_grad, tensor_sum.requires_grad)
    return tensor_sum

def streamify_pos_neg(quan_weight_tensor, bits):
    magic_number = 2 ** bits - 1
    positive_tensor = (torch.where(quan_weight_tensor > 0, quan_weight_tensor, 0) * magic_number).round().int()
    negative_tensor = (torch.where(quan_weight_tensor < 0, quan_weight_tensor.abs(), 0) * magic_number).round().int()
    # print(negative_tensor.grad, negative_tensor.is_leaf, negative_tensor.requires_grad)
    positive_arange = 2 ** torch.arange(bits - 1, -1, -1, device='cuda:0').int()
    negative_arange = 2 ** torch.arange(bits - 1, -1, -1, device='cuda:0').int()
    # pos_output = torch.sign(positive_tensor.bitwise_and(positive_arange).float())
    # neg_output = (-1 * negative_tensor.bitwise_and(negative_arange).float())

    output = positive_tensor.bitwise_and(positive_arange).float() + (-1 * negative_tensor.bitwise_and(negative_arange).float())
    return output


def streamify_pn_v2(tensor, bits):
    tensor_list = [tensor * (2 ** bits - 1)] * bits
    for i in range(bits):
        scalar = 2 ** (bits - i - 1)
        mask_pos = torch.where(tensor_list[i] > scalar, 1, 0).bool()
        mask_neg = torch.where(tensor_list[i] < -scalar, 1, 0).bool()
        for j, remaining_tensors in enumerate(tensor_list[i+1:]):
            tensor_list[(i+j+1)] = torch.where(mask_pos, tensor_list[(i+j+1)] - scalar, tensor_list[(i+j+1)])
            tensor_list[(i+j+1)] = torch.where(mask_neg, tensor_list[(i+j+1)] + scalar, tensor_list[(i+j+1)])
    output = torch.cat(tensor_list, -1).clamp(-1, 1)
    return output


def streamify_input(input_tensor, bits):
    magic_number = 2 ** bits - 1
    positive_tensor = (torch.where(input_tensor > 0, input_tensor, 0) * magic_number).round().int()
    positive_arange = 2 ** torch.arange(bits - 1, -1, -1, device='cuda:0').int()
    output = torch.sign(positive_tensor.bitwise_and(positive_arange).float())
    # output = positive_tensor.bitwise_and(positive_arange).float()
    return output


if __name__ == '__main__':
    gen_weight_vector_and_sum(bits=4)
