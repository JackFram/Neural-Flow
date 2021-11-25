import torch


def get_size(dtype):
    if dtype is torch.float32 or dtype is torch.float:
        return 4e-6
    elif dtype is torch.float64 or dtype is torch.double:
        return 8e-6
    elif dtype is torch.float16 or dtype is torch.half:
        return 2e-6
    elif dtype is torch.uint8 or dtype is torch.int8:
        return 1e-6
    elif dtype is torch.quint8 or dtype is torch.qint8:
        return 1e-6
    elif dtype is torch.int16 or dtype is torch.short:
        return 2e-6
    elif dtype is torch.int32 or dtype is torch.int:
        return 4e-6
    elif dtype is torch.int64 or dtype is torch.long:
        return 8e-6
    elif dtype is torch.bool:
        return 0.125e-6
    else:
        raise NotImplementedError(f"{dtype} not defined for get_size.")