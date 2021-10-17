from model import ResNet18
from netflow import *
from metric import TopologySimilarity
from opt import QuantizeOp
from dataset import get_dataset


if __name__ == "__main__":
    model = ResNet18()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    checkpoint = torch.load("./misc/checkpoint/ckpt_resnet18_cifar10.pth")
    model.load_state_dict(checkpoint['net'])
    model.eval()
    ds = get_dataset("cifar10")
    train_loader = ds.get_train_loader()
    test_loader = ds.get_test_loader()
    
    flow = FxInt(model)
    # flow.run(next(iter(train_loader)))

    for x, y in train_loader:
        x = x.to(device)
        flow.run(x)
        feature_list = flow.get_feature_list()
        name_list = flow.get_name_list()
        print(name_list)
        op = QuantizeOp(model)
        op.set_config(name_list)
        # op = PruningOp(model)
        print(op.operatable)

        metric = TopologySimilarity()
        for name in op.operatable:
            idx = name_list.index(name)
            print(name)
            print(metric.get_batch_score(feature_list[1], feature_list[idx]))
            # op.apply([name])
            #eval(op.mod_model, test_loader)
            #op.reset()

        break

    # TODO: evaluation and visualization

