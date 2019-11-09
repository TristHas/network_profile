import torch.nn as nn

from . import cal_op
from .helpers import train_model

DEFAULT_LTYPE = {nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.ReLU, nn.ReLU6, nn.AvgPool2d}

def select_layers(model, ltype=DEFAULT_LTYPE):
    """
        Filter the submodules of model according to ltype.
    """
    check_ltype = lambda x: type(x) in ltype 
    return list(filter(check_ltype, model.modules()))

def filter_mod_name(module):
    """
        Parse the module type name.
        nn.Module -> str
        Example: <class 'torch.nn.modules.conv.Conv2d'> -> Conv2d
    """
    return str(module.__class__).split(".")[-1].split("'")[0]

def isconv(layer_name):
    return layer_name.find("Conv") == 0

def t_closure_func(layer_stats):
    """
    """
    def get_layer_statistics(module, input, output):
        layer_name = filter_mod_name(module)
        ker = module.weight.shape[2:] if isconv(layer_name) else None
        info = [layer_name, input[0].shape, output.shape, ker]
        layer_stats.append(info)        
    return get_layer_statistics

def t_register_layer_hook(layers):
    """
    """
    handles, layer_stats = [], []
    for layer in layers:
        handles.append(layer.register_forward_hook(t_closure_func(layer_stats)))
    return handles, layer_stats

def t_profile_net(model, inp, layer_type=DEFAULT_LTYPE):
    """
    """
    fwd_time, bwd_time = t_profile_timings(model, inp)
    fw_flops, bw_flops, names, \
    in_size, out_size  = t_profile_theory(model, inp, layer_type)
    data = summarize_df(fwd_time, bwd_time, 
                        fw_flops, bw_flops, names, 
                        in_size, out_size)
    return data

def t_profile_theory(model, inp, layer_type=DEFAULT_LTYPE):
    """
    """
    layers = select_layers(model, layer_type)
    hands, info = t_register_layer_hook(layers)
    train_model(model, inp) #get info
    return t_summarize_layers_stats(info)

def t_summarize_layers_stats(info_collect):
    """
        Computes the number of fwd and bwd FLOP for each layer.
        Returns layer sizes, names and FLOPS in standardized lists.
        ____________________________
        Inputs:
            info_collect: list of list containing:
            layer name, inp size out, size and ks if convolution:
            [(name, in_size, out_size, [ker_size]), ... ]
        ____________________________
        Outputs:
            fw_flops    = list of layer fw_flops (int)
            bw_flops    = list of layer names (int)
            names       = list of layer names (str)
            input_size  = list of layer input_size  (tuple of int)
            output_size = list of layer output_size (tuple of int)
        
    """
    names, in_sizes, out_sizes, ker = list(zip(*info_collect))
    fw_flops, bw_flops = [],[]
    
    for name, out_size, in_size, k in zip(names, out_sizes, in_sizes, ker):
        func = getattr(cal_op, str("cal_" + name))
        if k is None:
            fw_flop = func(in_size, out_size)
            bw_flop = fw_flop 
        else:
            fw_flop = func(in_size, out_size, k)
            bw_flop = 2*fw_flop 
        
        fw_flops.append(fw_flop)
        bw_flops.append(bw_flop)
    return fw_flops, bw_flops, names, in_sizes, out_sizes