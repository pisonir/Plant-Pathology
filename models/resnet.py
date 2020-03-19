import torch.nn as nn
from torchvision.models import resnet18, resnet50
from functools import partial


def resnet_finetune(model, n_classes):
    """
    This function prepares resnet to be finetuned by:
    1) freeze the model weights
    2) cut-off the last layer and replace with a new one with the correct classes number
    """
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, n_classes)
    return model

# replace the resnet18 function
resnet50 = partial(resnet_finetune, resnet50(pretrained=True))