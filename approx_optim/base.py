import torch
import torch.nn as nn
import os


class BaseOp(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self.mod_model = None
        self.op_name = "quantize"

    def apply(self, name, *kargs):
        raise NotImplementedError

    def print_size(self):

        torch.save(self.model.state_dict(), "temp.p")
        print('Model Size before {} (MB):'.format(self.op_name), os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

        torch.save(self.model.state_dict(), "temp.p")
        print('Model Size after {} (MB):'.format(self.op_name), os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')