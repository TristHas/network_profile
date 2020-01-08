import numpy as np
import torch
import torch.nn.functional as F
from .load_func_map import get_func_map
from .config import data_dict

func_map = get_func_map(data_dict)

def conv3d_outshape(input, weight, stride, padding, dilation = 1):
    bs, cin, zin, hin, win = input.shape
    cout, cin, kz, kh, kw = weight.shape

    zout = conv3d_ZHW_size(zin, stride[0], padding[0], kz, dilation)
    hout = conv3d_ZHW_size(hin, stride[1], padding[1], kh, dilation)
    wout = conv3d_ZHW_size(win, stride[2], padding[2], kw, dilation)
    
    return (bs, cout, zout, hout, wout)

def conv3d_ZHW_size(x,s,p,k,d=1):
    return int(np.floor(((x + 2 * p - d * (k - 1) - 1)) / s) + 1)

def reshape_inp_inpgrad(tensor):
    bs,c,z,h,w = tensor.shape
    return tensor.contiguous().view(bs*c,1,z,h,w)

def newshape_weight(weight_shape, input_shape, grad_shape, stride,padding,dilation):
    if type(dilation) == int:
        dilation = (dilation, dilation, dilation)
    if type(stride) == int:
        stride = (stride, stride, stride)
    kz = conv3d_ZHW_size(input_shape[2], stride[0], padding[0], grad_shape[2],dilation[0])
    kh = conv3d_ZHW_size(input_shape[3], stride[1], padding[1], grad_shape[3],dilation[1])
    kw = conv3d_ZHW_size(input_shape[4], stride[2], padding[2], grad_shape[4],dilation[2])
    return (weight_shape[1], weight_shape[0], kz, kh, kw)
    
def return_correct_shape(weight_grad,weight_size):
    return weight_grad.transpose(0,1).narrow(
            2, 0, weight_size[2]).narrow(3, 0, weight_size[3]).narrow(
                4, 0, weight_size[4])

def get_implem(inp_shape, cin, cout, kernel, stride, padding, dilation, groups, bias):
    key = (inp_shape, cin, cout, kernel, stride, padding, dilation, groups, bias)
    return func_map[key]

def get_conv3d(inp_shape, cin, cout, kernel, stride, padding, dilation, groups, bias):
    
    key = (inp_shape, cin, cout, kernel, stride, padding, dilation, groups, bias)
    if not key in func_map:
        print("cannot find autotvm kernel for parameters:")
        print(f"inp_shape:{inp_shape}, cin:{cin}, cout:{cout}, kernel:{kernel}, stride:{stride}")
        print(f"padding:{padding}, dilation:{dilation}, groups:{groups}, bias:{bias}")
        return F.conv3d
    else:
        class MyConv3d(torch.autograd.Function):
            """
            """
            func = get_implem(inp_shape, cin, cout, kernel, stride, padding, dilation, groups, bias)
            @staticmethod
            def forward(ctx, input, weight, bias, stride, padding, dilation, groups = 1):
                """
                """
                ctx.constant = [MyConv3d.func["bw_data"], MyConv3d.func["bw_weight"], stride, padding, dilation, groups]

                o_shape = conv3d_outshape(input, weight, stride, padding, dilation)
                output = torch.zeros(size = o_shape, device = input.device)

                MyConv3d.func["fw"](input, weight, output)

                ctx.save_for_backward(input, weight)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                """
                """
                input, weight = ctx.saved_tensors
                
                data_grad_func, weight_grad_func, stride, padding, dilation, groups = ctx.constant
                
                input_g = torch.zeros(size = input.shape, device = input.device)
                
                new_weight_g_shape = newshape_weight(weight.shape,input.shape,grad_output.shape,dilation,padding,stride)
                weight_g = torch.zeros(size = new_weight_g_shape, device = weight.device)
                
                #computation
                data_grad_func(grad_output, weight, input_g)
                
                weight_grad_func(reshape_inp_inpgrad(input), reshape_inp_inpgrad(grad_output), weight_g)
                weight_g = return_correct_shape(weight_g, weight.shape)
                
                return input_g, weight_g, None, None, None, None, None, None
    
        return MyConv3d().apply