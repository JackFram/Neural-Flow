from .cv_dataloader import *


def get_dataset(name: str, args, root: str = '../data'):
    print('==> Preparing data..')
    if name == "cifar10":
        return Cifar10(root, args)
    if name == "imagenet":
        return ImageNet(os.path.join(root, "ImageNet"), args)
    if name == "imagenet16":
        return ImageNet16(os.path.join(root, "ImageNet16"), args)
