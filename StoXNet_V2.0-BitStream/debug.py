import torch
import torch.nn.functional as F
subarray_dimension = 256

def tensor_stats(tensor):
    min = torch.min(tensor).item()
    max = torch.max(tensor).item()
    mean = torch.mean(tensor).item()
    std_dev = torch.std(tensor).item()
    return ["Min: ", min,"Max: ", max,"Mean: ", mean, "Std: ", std_dev]

def find_empty_chunking(image_map, stride=(1, 1), padding=1, dilation=(1, 1)):
    kernel_width = kernel_height = 3
    # flattened_weights = torch.flatten(weights, 1)

    kernels = F.unfold(image_map, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding,
                       dilation=dilation)  # Dimensions = [batch size, in_chann * kernel_height * kernel_width, num kernels = image area] middle dim = lines up kernel vectors with weights

    num_chunks = (kernels.size(1) / subarray_dimension).__ceil__()
    kernel_list = torch.chunk(kernels, chunks=num_chunks, dim=1)
    # weight_list = torch.chunk(flattened_weights, chunks=num_chunks, dim=1)
    done = 0
    for i in range(len(kernel_list)):
        # working_weight = weight_list[i]
        working_kernel = kernel_list[i].transpose(-2, -1)
        if (working_kernel.mean()) == 0 and (working_kernel.max()) == 0 and (working_kernel.min()) == 0:
            print(kernel_list[i-1], kernel_list[i+1], kernel_list[i])
            print(tensor_stats(kernel_list[i-1]), tensor_stats(kernel_list[i+1]), tensor_stats(kernel_list[i]))
            print("!!!WE HAVE AN ERROR!!!")
            print(num_chunks, i, tensor_stats(working_kernel))
            done = 1
    if done == 1:
        exit()