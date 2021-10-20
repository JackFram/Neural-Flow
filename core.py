from model import ResNet18
from netflow import *
from metric import TopologySimilarity
from opt import QuantizeOp, PruningOp
from dataset import get_dataset
from cook.greedy import Greedy
from misc.eval import eval
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = ResNet18()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    checkpoint = torch.load("../data/checkpoint/ckpt_resnet18_cifar10.pth")
    model.load_state_dict(checkpoint['net'])
    model.eval()
    ds = get_dataset("cifar10")
    train_loader = ds.get_train_loader()
    test_loader = ds.get_test_loader()
    
    flow = FxInt(model)
    # flow.run(next(iter(train_loader)))

    feature_list = None
    name_list = None

    x, y = next(iter(train_loader))
    x = x.to(device)
    flow.run(x)
    feature_list = flow.get_feature_list()
    name_list = flow.get_name_list()

    ops = [PruningOp]

    metric = TopologySimilarity()

    chief = Greedy(
        model=model,
        ops=ops,
        metric=metric,
        flow=flow
    )
    for rate in np.arange(0, 1, 0.05):
        print("rate:%f "%rate)

        model = chief.run(rate=rate)

        print(eval(model.to(device), test_loader))

