from dataset import get_dataset
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

def get_cv_FIM(args, model, tokenizer, layer_name, logger, prefix=""):
    # get train loader
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    ds = get_dataset(args.data, args)
    train_loader = ds.get_train_loader()
    print("***** Getting Empirical Fisher Information Matrix *****")
    print(f"  Layer = {layer_name}")

    progress_bar = tqdm(range(args.train_batch_size))
    model.train()
    for _, batch in enumerate(train_loader):
        FIM = None
        for i in range(args.train_batch_size):
            image, target = batch[0][i].unsqueeze(0), batch[1][i].unsqueeze(0)
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # if torch.cuda.is_available():
            #     target = target.cuda(args.gpu, non_blocking=True)
            output = model(**image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            progress_bar.update(1)
            weight = model.get_submodule(layer_name).weight.grad.cpu().numpy().flatten()
            if hasattr(model.get_submodule(layer_name), "bias") and model.get_submodule(layer_name).bias is not None:
                bias = model.get_submodule(layer_name).bias.grad.cpu().numpy().flatten()
                param = np.concatenate([weight, bias], axis=0)
            else:
                param = weight
            if FIM is None:
                FIM = param**2
            else:
                FIM += param**2
        return FIM/args.train_batch_size