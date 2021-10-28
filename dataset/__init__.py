from .cv_dataloader import *


def get_dataset(name: str, root: str = '../data/ImageNet16'):
    print('==> Preparing data..')
    if name == "cifar10":
        return Cifar10(root)
    if name == "imagenet16":
        return ImageNet(root)
