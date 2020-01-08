import math
import torch
import torch.nn as nn
from .func import get_conv3d


class Conv3d(nn.Module):
    def __init__(self, cin, cout, kernel, 
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias = False):
        
        super(Conv3d, self).__init__()
        self.cin, self.cout, self.kernel, self.stride, \
        self.padding, self.dilation, self.groups, self.bias, = \
        cin, cout, kernel, stride, padding, dilation, groups, bias 
        
        self.weight = torch.nn.Parameter(torch.zeros(size = [cout, cin] + list(kernel)))
        self.reset_parameters()
        
    def reset_parameters(self):
        n = self.cin
        for k in self.kernel:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        key = (x.shape, self.cin, self.cout, self.kernel, self.stride, self.padding, self.dilation, self.groups, self.bias)
        conv = get_conv3d(*key)
        
        return conv(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)