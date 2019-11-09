import cal_op
import torch.nn as nn

DEFAULT_LTYPE = {nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, nn.ReLU, nn.ReLU6, nn.AvgPool2d}

def check_type(x,ltype=DEFAULT_LTYPE):
   
    layer_type = type(x)
    if layer_type in ltype:
        return layer_type 


def select_layers(model, ltype=DEFAULT_LTYPE):
    """
        Select submodules of model with type ltype.
        __________________
        Inputs:
            - model: nn.Module. model to select from
            - ltype: collection of layer types (sublcasses of nn.Module)
        __________________
        Outputs:
            - list of selected layers (nn.Module instances)
    """
    #check_type = type(lambda x: lambda x: type(x) in ltype)
    #check_type = lambda x: lambda x:type(x) in ltype
    return list(filter(check_type, model.modules()))




def filter_mod_name(module):
    """
        Parse xxx.xxx.xxx into xxx
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
        info = [layer_name, input[0].shape, output[0].shape, ker]
        layer_stats.append(info)        
    return get_layer_statistics

def t_register_layer_hook(layers):
    """
    """
    handles, layer_stats = [], []
    for layer in layers:
        handles.append(layer.register_forward_hook(t_closure_func(layer_stats)))
    return handles, layer_stats










# def search_layer(model, L):
#     """
    
#     """
#     for layer in list(model.children()):
#         if type(layer) not in [nn.Conv2d,nn.BatchNorm2d,nn.MaxPool2d,nn.ReLU,nn.ReLU6,nn.AvgPool2d]:
#             search_layer(layer, L)
#         else: 
#             L.append(layer)
                 
# def closure_func(info_collect):
#     """
#     """
#     info_collect = info_collect
#     def get_layer_info(module,input,output):
#         layer_name = str(module.__class__).split(".")[-1].split("'")[0]
#         if layer_name.find("Conv") == 0:
#             info = [layer_name, input[0].shape, output[0].shape,module.weight.shape]
#         else:
#             info = [layer_name, input[0].shape, output[0].shape]
#         info_collect.append(info)
#     return get_layer_info

# def register_layer_hook(layers, info_collect):
#     """
#     """
#     hands=[]
#     for layer in layers:
#         hands.append(layer.register_forward_hook(closure_func(info_collect)))
#     return hands

# def layer_info(info_collect):
#     """
#         Computes the number of fwd and bwd FLOP for each layer.
#         Returns layer sizes, names and FLOPS in standardized lists.
#         ____________________________
#         Inputs:
#             info_collect: list of list containing:
#             layer name, inp size out, size and ks if convolution:
#             [(name, in_size, out_size, [ker_size]), ... ]
#         ____________________________
#         Outputs:
#             fw_flops    = list of layer fw_flops (int)
#             bw_flops    = list of layer names (int)
#             names       = list of layer names (str)
#             input_size  = list of layer input_size  (tuple of int)
#             output_size = list of layer output_size (tuple of int)
        
#     """
#     fw_flops = []
#     bw_flops = []
#     names = []
#     input_size = []
#     output_size = []
    
#     for val in info_collect:
#         name = val[0]
        
#         func = "cal_" + name
#         cal_func = getattr(cal_op,str(func))
#         inpu = val[1]
#         input_size.append(inpu)
#         outpu = val[2]
#         output_size.append(outpu)
#         k = None
#         if len(val) == 4: #conv
#             k = val[3]
#             flops = cal_func(inpu,outpu,(k[2:]))
#             fw_flops.append(flops)
#             bw_flops.append(2*flops)
#             if k[1] == 1:          #depth wise conv
#                 name = name + "_dw"
#         else:
#             flops = cal_func(inpu,outpu)
#             fw_flops.append(flops)
#             bw_flops.append(flops)
#         names.append(name)
#     return fw_flops, bw_flops, names, input_size, output_size