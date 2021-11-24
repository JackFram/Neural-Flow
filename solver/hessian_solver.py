from .base_solver import *


class OneShotHessianSolver(BaseSolver):
    def __init__(self, net, ops):
        super().__init__(net, ops)

    def get_profile(self, layer_name):
        for Op in self.ops:
            op = Op(self.net)
            if layer_name in op.operatable:
                _, diff = op.apply(layer_name, amount=0.5, with_diff=True)
        return

    def get_solution(self, storage, latency):
        return