import torch
import torchvision.models as models
import torchvision.transforms as T

vgg16 = models.vgg16(pretrained=False)