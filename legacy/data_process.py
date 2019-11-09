import pandas as pd
import numpy as np
from autograd_profile import *
from layer_info import *

def select_attributes(events, attr = "cuda_time", operation = "conv2d"):
    """
    """
    return list(map(lambda x:getattr(x, attr), filter(lambda x:x.name==operation ,events)))

def list_operations(events):
    """
    """
    return set(map(lambda x:x.name, events))

def remove_rep(x,repetitive_operation):
    """
    """
    for op in repetitive_operation: 
        for index in x.index:
            if index == op:
                x=x.drop(op)   
    return x

def parse_operation_time(events):
    """
    """
    return pd.Series({operation: select_attributes(events, attr='cuda_time', operation=operation)\
                                 for operation in list_operations(events)})




def average_over_iter(x, n=1):
    """
    """
    n = len(x) // n
    return list(np.mean(np.asarray(x).reshape(-1,n),axis=0))




"""
num: iteration number
layer_map: The kernel function of the corresponding fw and bw 
"""
num=1

def layer_kernels():
    layer_map = {"name":["Conv2d","Conv2d_dw","BatchNorm2d","ReLU","MaxPool2d","ReLU6","AvgPool2d"],
                   "fw":["cudnn_convolution","thnn_conv_depthwise2d_forward","batch_norm","relu_","max_pool2d","hardtanh_","avg_pool2d"],
                   "bw":["cudnn_convolution_backward","thnn_conv_depthwise2d_backward","CudnnBatchNormBackward","ReluBackward1","MaxPool2DWithIndicesBackward","HardtanhBackward1","avg_pool2d_backward"]}

    df = pd.DataFrame(index=layer_map["name"],columns=["fw","bw"])
    df["fw"] = layer_map["fw"]
    df["bw"] = layer_map["bw"]
    return df

DEFAULT_LAYER_KERNEL = layer_kernels()


def t_summarize_layers_timing(cuda_r, lk_map=DEFAULT_LAYER_KERNEL, nrep=1):
    """
        summarize the cuda profiler infos.
        _____________
        Inputs:
         - 
          - 
            - 
        _____________
        Outputs
        
    """
    results = parse_operation_time(cuda_r.function_events)
    
    fwd_ops = lk_map["fw"][lk_map["fw"].isin(results.index)]
    bwd_ops = lk_map["bw"][lk_map["bw"].isin(results.index)]
    
    fwd_op_time = []
    bwd_op_time = []

    for fwd_op, bwd_op in zip(fwd_ops, bwd_ops):
        fw_time = average_over_iter(results[str(fwd_op)], nrep)
        bw_time = average_over_iter(results[str(bwd_op)], nrep)
        
        fwd_op_time.extend(fw_time)
        bwd_op_time.extend(bw_time)
    
    return fwd_op_time, bwd_op_time

def summarize_df(fwd_time, bwd_time, 
                 fw_flops, bw_flops, 
                 names, in_size, out_size):
    """
        Return a pandas.DataFrame summarizing the input sequences
    """
    return pd.DataFrame({"layer":names,
                         "fw_operation":fw_flops,
                         "forward_time":fwd_time,
                         "forward_effi": np.array(fw_flops)/np.array(fwd_time),
                         "bw_operation": bw_flops,
                         "backward_time":bwd_time,
                         "backward_effi":np.array(bw_flops)/np.array(bwd_time),
                         "bw_time/fw_time":np.array(bwd_time)/np.array(fwd_time),
                         "input_size":in_size,
                         "output_size":out_size
                    })

def train_model(model, inp):
    out = model(inp)
    out.sum().backward()

def t_profile_theory(model, inp):
    """
    """
    layers = select_layers(model)
    hands, info = t_register_layer_hook(layers)
    train_model(model, inp) #get info
    return t_summarize_layers_stats(info)

def t_profile_timings(model, inp):
    """
    """
    cuda_r, _ = run_autograd_prof(train_model, model, inp)
    return t_summarize_layers_timing(cuda_r, lk_map=DEFAULT_LAYER_KERNEL, nrep=1)

def t_summarize_layers_stats(info_list):
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
    names, in_sizes, out_sizes , ker = list(zip(*info_list))
    fw_flops, bw_flops = [],[]
    
    for name, out_size, in_size, k in zip(names, out_sizes, in_sizes, ker):
        func = getattr(cal_op, str("cal_" + name))
        fw_flop = func(in_size, out_size, k)
        bw_flop = fw_flop if k is None else 2*fw_flop
        
        fw_flops.append(fw_flop)
        bw_flops.append(bw_flop)
    return fw_flops, bw_flops, names, in_sizes, out_sizes