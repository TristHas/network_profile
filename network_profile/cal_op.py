import numpy as np
import torch.nn as nn

def mac_Conv2d(*para):
    input_shape = para[0]
    output_shape = para[1]
    kernel = np.prod(list(para[2]))
    #bs = output_shape[0]
    mac = np.prod(input_shape) + np.prod(output_shape) + (kernel)
    return mac

def mac_BatchNorm2d(*para):
    input_shape = para[0]
    output_shape = para[1]
    mac = np.prod(input_shape) + np.prod(output_shape)
    return mac

def mac_ReLU6(*para):
    return mac_BatchNorm2d(*para)

def mac_ReLU(*para):
    return mac_BatchNorm2d(*para)

def mac_MaxPool2d(*para):
    return mac_BatchNorm2d(*para)

def mac_Conv3d(*para):
    return mac_Conv2d(*para)

def mac_BatchNorm3d(*para):
    return mac_BatchNorm2d(*para)

def mac_MaxPool3d(*para):
    return mac_BatchNorm2d(*para)

# def cal_Conv_Mac(*para):
#     input_shape = para[0]
#     output_shape = para[1]
    
#     kernel = np.prod(list(para[2]))
    
#     c_in = input_shape[1]
#     c_out = output_shape[0]

#     mac = np.prod(input_shape) + np.prod(output_shape) + (kernel * c_in * c_out)
#     return mac

# def cal_BN_Mac(*para):
#     input_shape = para[0]
#     output_shape = para[1]
    
#     c_in = input_shape[1]
#     c_out = output_shape[-3]

#     mac = np.prod(input_shape) + np.prod(output_shape) 
    
#     return mac

def cal_Conv3d(*para):
    output_shape = para[1]
    
    HWZ = np.prod(output_shape[2:])
    bs = output_shape[0]
    kernel = np.prod(para[2])
    
    flop = kernel * HWZ * bs
    return flop

def cal_Conv2d(*para):
    output_shape = para[1]

    bs = output_shape[0]
    kernel = np.prod(para[2])
    H = output_shape[-1]
    W = output_shape[-2]
    
    flop = kernel * H * W * bs
    return flop

def cal_BatchNorm2d(*para, name='batch_norm'):
    return np.prod(para[0])  

def cal_BatchNorm3d(*para):
    input_shape = para[0]
    bs = input_shape[0]
    
    c_in = input_shape[1]
    Z = input_shape[2]
    Y = input_shape[3]
    X = input_shape[4]
    flops = bs * c_in * X * Y * Z
    return flops

def cal_ReLU(*para, name='relu'):
    return np.prod(para[0])  

def cal_ReLU6(*para, name='relu'):
    return cal_ReLU(*para)

def cal_MaxPool2d(*para, name='maxpool2d'):
    return np.prod(para[0])  

def cal_AvgPool2d(*para):
    return cal_MaxPool2d(*para)

def cal_MaxPool3d(*para):
    return cal_MaxPool2d(*para)