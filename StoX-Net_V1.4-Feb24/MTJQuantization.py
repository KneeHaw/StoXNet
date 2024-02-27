from functionsbase import *


class MTJInstance(Function):
    @staticmethod
    def forward(ctx, input_tens, sensitivity, pos_only):
        # input_tens_tanh = torch.tanh(sensitivity * input_tens)
        input_tens_tanh = 2 / (1 + torch.e ** (-3.59813326*(input_tens-.00821371))) - 1
        rand_tens = ((2 * torch.rand_like(input_tens_tanh, device='cuda:0')) - 1)
        mask1 = input_tens_tanh > rand_tens
        mask2 = input_tens == 0
        if pos_only:
            out = 1 * (mask1.type(torch.float32))
        else:
            out = (1 * (mask1.type(torch.float32)) - (1 - mask1.type(torch.float32))) * (1 - mask2.type(torch.float32))

        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def gen_weight_vector_and_sum(tensor, bits):
    vector = (2 ** torch.arange(0, bits, 1, device='cuda'))
    tensor_sum = torch.stack(torch.tensor_split(tensor, bits, dim=1), dim=-1) * vector / (2 ** bits - 1)
    tensor_sum = tensor_sum.sum(-1)
    return tensor_sum


def gen_image_vector_and_sum(tensor, bits):
    vector = (2 ** torch.arange(bits, device='cuda'))
    tensor_sum = torch.stack(torch.split(tensor, bits, dim=-1)) * vector / (2 ** bits - 1)
    tensor_sum = tensor_sum.permute(1, 2, 0, 3).sum(-1)
    return tensor_sum