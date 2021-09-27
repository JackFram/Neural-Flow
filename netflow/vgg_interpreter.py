from netflow.net_interpreter import NetIntBase
import torch.nn as nn


class VggInt(NetIntBase):
    def __init__(self, net):
        super().__init__(net=net)
        idx = 0
        for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d):
                self.idx_2_name[idx] = name
                idx += 1

    def sequentialize(self):
        self.seq_net = self.net.features

