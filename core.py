from model import ResNet18
from netflow import *
from metric import TopologySimilarity
from opt import QuantizeOp
from dataset import get_dataset
from cook.greedy import Greedy
from misc.eval import eval


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

    for x, y in train_loader:
        x = x.to(device)
        flow.run(x)
        feature_list = flow.get_feature_list()
        name_list = flow.get_name_list()
        break

    ops = [QuantizeOp]

    metric = TopologySimilarity()

    chief = Greedy(
        model=model,
        ops=ops,
        metric=metric,
        flow=flow
    )

    model = chief.run(rate=1)

    print(eval(model.to('cpu'), test_loader))

