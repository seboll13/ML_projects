import torch
from torch import nn

from architectures import resnet

def generate_model():
    res_net = resnet.resnet101(num_classes=120)

    return res_net