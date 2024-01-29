from functionsbase import *
from debug import tensor_stats


class WeightQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, k, t, bits):
        ctx.save_for_backward(input_tens, k, t)
        if bits > 1:
            input_tens = input_tens * (2 ** (bits - 1))
            out = torch.where(input_tens > 0, input_tens.ceil(), input_tens)
            out = torch.where(input_tens < 0, input_tens.floor(), out)
            out = torch.where(input_tens == 0, input_tens, out) / (2 ** (bits - 1))
        else:
            out = torch.where(input_tens < 0, torch.floor(input_tens), input_tens)
            out = torch.where(out > 0, torch.ceil(out), out)
        out = torch.clamp(out, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_tens, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input_tens * t), 2)) * grad_output
        return grad_input, None, None, None


def create_weight_sliced_adcless(tensor, bits, slice_precision, qn, qp):  # LSB to MSB stack dim, takes in int
    bit_stream = []
    tensor = tensor.clamp(qn, qp)
    tensor_negatives = torch.where(tensor < 0, -1, 1)
    tensor = tensor.abs()
    for i in range(int(bits / slice_precision)):
        temp_tensor = tensor / (2 ** slice_precision) ** i
        temp_tensor = adcless_floor().apply(temp_tensor)
        bit_stream.append(tensor_negatives * temp_tensor)
    bit_stream = torch.cat(bit_stream, 0)
    bit_stream = adcless_remainder().apply(bit_stream, 2 ** slice_precision)
    return bit_stream
