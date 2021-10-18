from .base import *
import numpy as np


class Greedy(CookBase):
    def __init__(self, model: nn.Module, ops: List[BaseOp], metric: Metric, flow: NetIntBase, rate=0):
        super().__init__(model=model, ops=ops, metric=metric, flow=flow, rate=rate)

    def run(self) -> nn.Module:
        model = self.model
        if self.metric is None:
            for op in self.ops:
                candidate = op.operatable
                sample = np.random.choice(candidate, size=int(self.rate*len(candidate)), replace=False)
                model = op.apply(sample)
            return model

        else:
            name_list = self.flow.get_name_list()
            feature_list = self.flow.get_feature_list()
            for Op in self.ops:
                op = Op(model)
                op.set_config()
                candidate = op.operatable
                score_list = []
                for name in candidate:
                    idx = name_list.index(name)
                    score_list.append(self.metric.get_batch_score(feature_list[idx - 1], feature_list[idx]))
                if isinstance(op, QuantizeOp):
                    size = int(self.rate*len(candidate))
                    sorted_index = list(reversed(np.argsort(score_list)))
                    sample = [candidate[sorted_index[i]] for i in range(size)]
                    model = op.apply(sample)
            return model
