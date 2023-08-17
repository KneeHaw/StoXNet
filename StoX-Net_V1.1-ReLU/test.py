import torch

def print_stats(tensor):
    print(torch.min(tensor), torch.max(tensor))


num_bits = 4
magic_number = 2 ** num_bits - 1
magic2 = magic_number/2

# print(magic_number, magic2)

num_sample = torch.rand(100)
print_stats(num_sample)
input_tens = (2 * torch.rand(100)) - 1
print_stats(input_tens)
input_tens = (torch.tanh(8 * (input_tens - .5)) + 1) / 2
print_stats(input_tens)
exit()
num_sample *= magic_number

num_sample = torch.where(num_sample < magic2, torch.floor(num_sample), num_sample)
num_sample = torch.where(num_sample > magic2, torch.ceil(num_sample), num_sample)

print(num_sample)
print_stats(num_sample)

num_sample /= magic_number

print_stats(num_sample)