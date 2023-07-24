import torch

tensor1 = torch.ones((128, 16, 1024))
tensor2 = torch.ones(16)
a = tensor1
b = tensor2
out = a - b
print(out, a.size(), b.size())
