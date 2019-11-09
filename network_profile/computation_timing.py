import pandas as pd
import numpy as np
from torch.autograd import profiler
from .helpers import train_model, select_layers, DEFAULT_LTYPE

def layer_kernels():
    layer_map = {"name":["Conv2d","Conv2d_dw","BatchNorm2d","ReLU","MaxPool2d","ReLU6","AvgPool2d"],
                   "fw":["cudnn_convolution","thnn_conv_depthwise2d_forward","batch_norm","relu_","max_pool2d","hardtanh_","avg_pool2d"],
                   "bw":["cudnn_convolution_backward", "thnn_conv_depthwise2d_backward", "CudnnBatchNormBackward", "ReluBackward1", 
                         "MaxPool2DWithIndicesBackward"," HardtanhBackward1",  "avg_pool2d_backward"]}

    df = pd.DataFrame(index=layer_map["name"],columns=["fw","bw"])
    df["fw"] = layer_map["fw"]
    df["bw"] = layer_map["bw"]
    return df

DEFAULT_LAYER_KERNEL = layer_kernels()

def average_over_iter(x, n=1):
    """
    """
    n = len(x) // n
    return list(np.mean(np.asarray(x).reshape(-1,n),axis=0))

def select_attributes(events, attr = "cuda_time", operation = "conv2d"):
    """
    """
    return list(map(lambda x:getattr(x, attr), filter(lambda x:x.name==operation ,events)))

def list_operations(events):
    """
    """
    return set(map(lambda x:x.name, events))

def parse_operation_time(events):
    """
    """
    return pd.Series({operation: select_attributes(events, attr='cuda_time', operation=operation)\
                                 for operation in list_operations(events)})

def run_autograd_prof(func,*para):
    with profiler.profile(use_cuda = True) as prof:
        func_return = func(*para)
    return prof, func_return

def t_summarize_layers_timing(cuda_r, lk_map=DEFAULT_LAYER_KERNEL, nrep=1):
    """
        summarize the cuda profiler infos.
        _____________
        Inputs:
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

def meta(results, num):
    """
    """
    total_time = results.apply(sum)
    repetition = results.apply(len)
    rep_per_iter = repetition // num
    correct = (repetition % num)==0
    
    return pd.DataFrame({"time":total_time, 
                         "rep":rep_per_iter})[correct].sort_values(by="time", ascending=False)

def t_profile_timings(model, inp):
    """
    """
    # Warm-up
    train_model(model, inp)
    # Profile
    cuda_r, _ = run_autograd_prof(train_model, model, inp)
    # Summarize results
    return t_summarize_layers_timing(cuda_r, lk_map=DEFAULT_LAYER_KERNEL, nrep=1)

