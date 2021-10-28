from .cv_dataloader import *


def get_dataset(name: str, root: str = '../data'):
    print('==> Preparing data..')
    if name == "cifar10":
        return Cifar10(root)
    if name == "imagenet16":
        return ImageNet16(os.path.join(root, "ImageNet16"))
