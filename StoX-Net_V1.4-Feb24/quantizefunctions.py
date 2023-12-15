from torch.autograd import Function
import torch
from debug import tensor_stats


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


class WeightQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, bits):
        ctx.save_for_backward(input_tens, k, t)
        magic_number_weights = 2 ** bits - 1
        out = input_tens * magic_number_weights

        if bits > 1:
            out = out.round()
        else:
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


def create_input_stream(tensor, bits):  # Takes -1, 1 input and binarizes it accordingly. Should be gradient safe?
    magic_number = 2 ** bits - 1
    tensor = torch.round(tensor * magic_number)
    negative_signs = torch.where(tensor < 0, -1, 1).unsqueeze(-1)
    arranged = (1 / (2 ** torch.arange(bits, device='cuda')).unsqueeze(0))
    bit_stream = torch.floor(torch.matmul(tensor.abs().clamp(0, magic_number).unsqueeze(-1), arranged).fmod(2)) * negative_signs
    bit_stream = bit_stream.reshape(bit_stream.size(0), bit_stream.size(1), -1)
    return bit_stream


def gen_weight_vector_and_sum(tensor, bits):
    scalar = 2 ** bits - 1
    vector = (2 ** torch.arange(bits - 1, -1, -1)).cuda().unsqueeze(-1).repeat(1, int(tensor.size(1)/bits)).flatten() / scalar
    tensor_split = torch.tensor_split(tensor * vector.view(1, -1, 1), sections=bits, dim=1)
    tensor_stack = torch.stack(tensor_split, dim=-1)
    tensor_sum = torch.sum(tensor_stack, dim=-1)
    return tensor_sum


def gen_image_vector_and_sum(tensor, bits):
    arranged = (2 ** torch.arange(bits, device='cuda'))
    tensor_sum = torch.stack(torch.split(tensor, bits, dim=-1))
    tensor_sum *= arranged
    tensor_sum = tensor_sum.permute(1, 2, 0, 3).sum(-1) / (2 ** bits - 1)
    return tensor_sum


if __name__ == '__main__':
    print(create_input_stream(torch.tensor([18. / 15, 15. / 15, 3. / 15, -5. / 15], device='cuda'), 4))
