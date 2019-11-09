import numpy as np
import pandas as pd
from torch.autograd import profiler
import torch.nn as nn

repetitive_operation= ['cudnn_convolution_backward','convolution','_convolution','cudnn_convolution','cudnn_batch_norm_backward',
                       'native_batch_norm_backward','native_batch_norm',
                       '_batch_norm_impl_index','cudnn_batch_norm','max_pool2d_with_indices_backward','max_pool2d_with_indices',
                       'threshold_backward',"elu_backward",
                       'upsample_trilinear3d_backward','max_pool3d_with_indices_backward','max_pool3d_with_indices']

def run_autograd_prof(func,*para):
    print("profiling your model")
    with profiler.profile(use_cuda = True) as prof:
        func_return = func(*para)
    print("finish")
    return prof, func_return
    
def cuda_time_total(autograd_prof):
    return sum([event.cuda_time_total for event in autograd_prof.function_events])

def select_attributes(events, attr = "cuda_time", operation = "conv2d"):
    return list(map(lambda x:getattr(x, attr), filter(lambda x:x.name==operation ,events)))

def list_operations(events):
    return set(map(lambda x:x.name, events))

def remove_rep(x,repetitive_operation):
    for op in repetitive_operation: 
        for index in x.index:
            if index == op:
                x=x.drop(op)   
    return x
def get_operation_time(events,sum_results=True):
    results = {}
    for operation in list_operations(events):
        cuda_times = select_attributes(events,attr='cuda_time', operation=operation)
        if(sum_results):
            results[operation] = sum(cuda_times)
        else:
            results[operation] = cuda_times
    return results

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

def average_over_iter(x, n):
    """
    """
    return np.mean(np.asarray(x).reshape(-1,n),axis=0)
