import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rand_tens = (3 * torch.rand(10000)) - 1.5

tanh = torch.tanh(rand_tens)
sign = torch.sign(rand_tens)
grad = (1 - torch.pow(tanh, 2))

plt.scatter(rand_tens, tanh, label="swprob(x) = tanh(4x)")
plt.scatter(rand_tens, sign, label="sign(x) = -1 if x<0 \nor 1 if x>0")
plt.scatter(rand_tens, grad, label="approx_grad(x) =\n(1 -(tanh(4x)^2))")
plt.legend()
plt.show()
