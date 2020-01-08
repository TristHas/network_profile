import fast3d
import torch.nn as nn
import torch

DEFAULT_LTYPE = {nn.Conv2d, nn.Conv3d, nn.BatchNorm2d, nn.BatchNorm3d, nn.MaxPool2d, nn.MaxPool3d, nn.ReLU, nn.ReLU6, nn.AvgPool2d, fast3d.module.Conv3d}

def select_layers(model, ltype=DEFAULT_LTYPE):
    """
        Filter the submodules of model according to ltype.
    """
    check_ltype = lambda x: type(x) in ltype 
    return list(filter(check_ltype, model.modules()))

def train_model(model, inp):
    """
    """
    out = model(inp)
    grad = torch.randn_like(out)
    out.backward(grad)