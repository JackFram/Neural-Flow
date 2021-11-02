import os
import torch
import numpy as np


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def avg_deviation(arr1, arr2):
    # arr shape (out, in)
    dev = np.linalg.norm(arr1-arr2, axis=1)
    return dev.mean()
