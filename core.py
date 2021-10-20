from model import ResNet18
from netflow import *
from metric import TopologySimilarity
from opt import QuantizeOp, PruningOp
from dataset import get_dataset
from cook.greedy import Greedy
from misc.eval import eval
import numpy as np
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
>>>>>>> 05c356d0c5d7e66942574fd5f9278d08eee192d7


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

<<<<<<< HEAD
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


=======
    results = []
    for i in np.arange(0, 0.8, 0.1):
        chef = Greedy(
            model=model,
            ops=ops,
            metric=metric,
            flow=flow
        )
        model = chef.run(rate=i)
        result = eval(model.to('cpu'), test_loader)
        print(result[0])
        results.append([i, result[0]])
        
    plt.plot(*np.array(results).T)
    plt.savefig("out/plot.png")
>>>>>>> 05c356d0c5d7e66942574fd5f9278d08eee192d7

