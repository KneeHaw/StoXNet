import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# num_list = torch.tensor([1.37, 2.35, 1.2, -.08, .14, 4.53, -2.24, .86, -1.02])
# old_range = torch.max(num_list) - torch.min(num_list)
# new_range = 255
# new_list = (((num_list - torch.min(num_list)) * new_range) / (old_range * 1.25))
# print(new_list)
# exit(0)

StoX_4_16 = "4w4a/StoX4w4aLog.txt"
StoX_4_64 = "4w4a/StoX4w4a64Log.txt"
StoX_4_256 = "4w4a/StoX4w4a256Log.txt"

StoX_2_16 = "2w2a250/StoX2w2aLog.txt"
StoX_2_64 = "2w2a250/StoX2w2a64Log.txt"
StoX_2_256 = "2w2a250/StoX2w2a256Log.txt"

StoX_1_16 = "1w1a309/StoX1w1aLog.txt"
StoX_1_64 = "1w1a309/StoX1w1a64Log.txt"
StoX_1_256 = "1w1a309/StoX1w1a256Log.txt"

StoX_4_16_acc = np.loadtxt(StoX_4_16, dtype=float, delimiter=', ')[:, 3]
StoX_4_64_acc = np.loadtxt(StoX_4_64, dtype=float, delimiter=', ')[:, 3]
StoX_4_256_acc = np.loadtxt(StoX_4_256, dtype=float, delimiter=', ')[:, 3]

StoX_2_16_acc = np.loadtxt(StoX_2_16, dtype=float, delimiter=', ')[:, 3]
StoX_2_64_acc = np.loadtxt(StoX_2_64, dtype=float, delimiter=', ')[:, 3]
StoX_2_256_acc = np.loadtxt(StoX_2_256, dtype=float, delimiter=', ')[:, 3]

StoX_1_16_acc = np.loadtxt(StoX_1_16, dtype=float, delimiter=', ')[:, 3]
StoX_1_64_acc = np.loadtxt(StoX_1_64, dtype=float, delimiter=', ')[:, 3]
StoX_1_256_acc = np.loadtxt(StoX_1_256, dtype=float, delimiter=', ')[:, 3]

# data = [StoX_4_16_data, StoX_4_64_data, StoX_4_256_data, StoX_2_16_data, StoX_2_64_data, StoX_1_16_data, StoX_1_64_data]
def print_max(array):
    print(np.max(array))

print_max(StoX_4_16_data)
print_max(StoX_4_64_data)
print_max(StoX_4_256_data)
print_max(StoX_2_16_data)
print_max(StoX_2_64_data)
print_max(StoX_1_16_data)
print_max(StoX_1_64_data)
#print_max(StoX_4_16_data)
#print_max(StoX_4_16_data)
exit()





plt.plot(epochs, StoX_train, label="StoX Train Acc")
plt.plot(epochs, ADC_train, label="ADC Train Acc")
plt.legend()
plt.show()




ADC_Log = "ADC_Log.txt"
StoX_Log = "StoX_Log.txt"

StoX_data = np.loadtxt(StoX_Log, dtype=float, delimiter=', ')
ADC_data = np.loadtxt(ADC_Log, dtype=float, delimiter=', ')
print(StoX_data)
epochs = StoX_data[:, 0]
StoX_loss = StoX_data[:, 1]
StoX_train = StoX_data[:, 2]
StoX_test = StoX_data[:, 3]

ADC_loss = ADC_data[:, 1]
ADC_train = ADC_data[:, 2]
ADC_test = ADC_data[:, 3]

plt.plot(epochs, StoX_train, label="StoX Train Acc")
plt.plot(epochs, ADC_train, label="ADC Train Acc")
plt.legend()
plt.show()

plt.plot(epochs, StoX_loss, label="StoX Loss")
plt.plot(epochs, ADC_loss, label="ADC Loss")
plt.legend()
plt.show()

plt.plot(epochs, StoX_test, label="StoX Test Acc")
plt.plot(epochs, ADC_test, label="ADC Test Acc")
plt.legend()
plt.show()
