import torch
from torch import nn

from architectures import resnet

def generate_model(opt):
    assert opt.mode in ['score', 'feature']
    if opt.mode == 'score':
        last_fc = True
    elif opt.mode == 'feature':
        last_fc = False
    
    assert opt.model_name = 'resnet'
    assert opt.model_depth = 101
    model = resnet.resnet101(
        num_classes=opt.n_classes, 
        shortcut_type=opt.resnet_shortcut,
        sample_size=opt.sample_size, 
        sample_duration=opt.sample_duration,
        last_fc=last_fc
    )

    return model