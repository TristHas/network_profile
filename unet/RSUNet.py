import collections
from collections import OrderedDict
from itertools import repeat
import math
from fast3d import Conv3d
import torch
from torch import nn
from torch.nn import functional as F
MODE = "tvm"

# Number of feature maps.
# nfeatures = [24,32,48,72,104,144]
nfeatures = [32,64,128,192]

# Filter size.
sizes = [(3,3,3)] * len(nfeatures)

# In/out embedding.
embed_ks = (1,5,5)
embed_nin = nfeatures[0]
embed_nout = embed_nin


def _ntuple(n):
    """
    Copied from PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


def pad_size(kernel_size, mode):
    assert mode in ['valid', 'same', 'full']
    ks = _triple(kernel_size)
    if mode == 'valid':
        pad = (0,0,0)
    elif mode == 'same':
        assert all([x %  2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad


def batchnorm(out_channels, use_bn, momentum=0.001, track=False):
    if use_bn:
        layer = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=momentum, track_running_stats=track)
    else:
        layer = lambda x: x
    return layer

def residual_sum(x, skip, residual):
    return x + skip if residual else x
    
    
class RSUNet(nn.Module):
    """
        Residual Symmetric U-Net (RSUNet).
    """
    def __init__(self, aff, depth,
                 residual=True, upsample='bilinear', use_bn=True,
                 momentum=0.001, track=False, activation=F.elu):
        super(RSUNet, self).__init__()
        self.residual = residual
        self.upsample = upsample
        self.use_bn   = use_bn
        self.momentum = momentum

        in_channels = 1
        # Model depth (# scales == depth + 1).
        assert depth < len(nfeatures)
        self.depth = depth

        # Input feature embedding without batchnorm.
        self.embed_in = EmbeddingMod(in_channels, embed_nin, embed_ks, activation=activation)
        in_channels = embed_nin

        # Contracting/downsampling pathway.
        for d in range(depth):
            fs, ks = nfeatures[d], sizes[d]
            self.add_conv_mod(d, in_channels, fs, ks, track=track, activation=activation)
            self.add_max_pool(d+1, fs)
            in_channels = fs

        # Bridge.
        fs, ks = nfeatures[depth], sizes[depth]
        self.add_conv_mod(depth, in_channels, fs, ks, track=track, activation=activation)
        in_channels = fs

        # Expanding/upsampling pathway.
        for d in reversed(range(depth)):
            fs, ks = nfeatures[d], sizes[d]
            self.add_upsample_mod(d, in_channels, fs, track=track, activation=activation)
            in_channels = fs
            self.add_dconv_mod(d, in_channels, fs, ks, track=track, activation=activation)

        # Output feature embedding without batchnorm.
        self.embed_out = EmbeddingMod_out(in_channels, embed_nout, embed_ks, activation=activation)
        in_channels = embed_nout

        # Output by spec.
        self.output = OutputMod(in_channels, aff)

    def add_conv_mod(self, depth, in_channels, out_channels, kernel_size, track=False, activation=F.elu):
        name = 'convmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels, kernel_size,
                         residual=self.residual, use_bn=self.use_bn,
                         momentum=self.momentum, track=track, activation=activation)
        self.add_module(name, module)

    def add_dconv_mod(self, depth, in_channels, out_channels, kernel_size, track=False, activation=F.elu):
        name = 'dconvmod{}'.format(depth)
        module = ConvMod(in_channels, out_channels, kernel_size,
                         residual=self.residual, use_bn=self.use_bn,
                         momentum=self.momentum, track=track, activation=activation)
        self.add_module(name, module)

    def add_max_pool(self, depth, in_channels, down=2): ####
        name = 'maxpool{}'.format(depth)
        module = nn.MaxPool3d(down)
        self.add_module(name, module)

    def add_upsample_mod(self, depth, in_channels, out_channels, up=2, track=False, activation=F.elu):  ####
        name = 'upsample{}'.format(depth)
        module = UpsampleMod(in_channels, out_channels, up=up,
                             mode=self.upsample, use_bn=self.use_bn,
                             momentum=self.momentum, track=track, activation=activation)
        self.add_module(name, module)

    def forward(self, x):
        # Input feature embedding without batchnorm.
        x = self.embed_in(x)
        # Contracting/downsmapling pathway.
        skip = []
        for d in range(self.depth):
            convmod = getattr(self, 'convmod{}'.format(d))
            maxpool = getattr(self, 'maxpool{}'.format(d+1))
            x = convmod(x)
            skip.append(x)
            x = maxpool(x)
        #torch.cuda.synchronize()
        # Bridge.
        bridge = getattr(self, 'convmod{}'.format(self.depth))
        x = bridge(x)
        #torch.cuda.synchronize()
        # Expanding/upsampling pathway.
        for d in reversed(range(self.depth)):
            upsample = getattr(self, 'upsample{}'.format(d))
            dconvmod = getattr(self, 'dconvmod{}'.format(d))
            x = dconvmod(upsample(x, skip[d]))
        #torch.cuda.synchronize()
        # Output feature embedding without batchnorm.
        x = self.embed_out(x)
        out = self.output(x)
        return out
        
        
class Conv(nn.Module):
    """
        3D convolution w/ MSRA init.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, sizes=False, bias=False):
        super(Conv, self).__init__()
        if MODE=="tvm":
            self.conv = Conv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=None)
        else:
            self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False)

        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ConvMod(nn.Module):
    """
    Convolution module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu, residual=True, use_bn=True,
                 momentum=0.001, track=False):
        super(ConvMod, self).__init__()
        # Convolution params.
        ks = _triple(kernel_size)
        st = (1,1,1)
        pad = pad_size(ks, 'same')
        bias = not use_bn
        # Convolutions.
        self.conv1 = Conv(in_channels,  out_channels, ks, st, pad, bias)
        self.conv2 = Conv(out_channels, out_channels, ks, st, pad, bias)
        self.conv3 = Conv(out_channels, out_channels, ks, st, pad, bias)
        # BatchNorm.
        self.bn1 = batchnorm(out_channels, use_bn, momentum=momentum, track=track)
        self.bn2 = batchnorm(out_channels, use_bn, momentum=momentum, track=track)
        self.bn3 = batchnorm(out_channels, use_bn, momentum=momentum, track=track)
        # Activation function.
        self.activation = activation
        # Residual skip connection.
        self.residual = residual

    def forward(self, x):
        # Conv 1.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        skip = x
        # Conv 2.
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # Conv 3.
        x = self.conv3(x)
        x = residual_sum(x, skip, self.residual)
        x = self.bn3(x)
        return self.activation(x)

class UpsampleMod(nn.Module):
    """
    Transposed Convolution module.
    """
    def __init__(self, in_channels, out_channels, up=(1,2,2), mode='bilinear',
                 activation=F.elu, use_bn=True, momentum=0.001, track=False):
        super(UpsampleMod, self).__init__()
        # Convolution params.
        ks = (1,1,1)
        st = (1,1,1)
        pad = (0,0,0)
        bias = False#True
        # Upsampling.
        if mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=up, mode='trilinear')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=up, mode='nearest')
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'transpose':
            raise NotImplementedError
        else:
            assert False, "unknown upsampling mode {}".format(mode)
        # BatchNorm and activation.
        self.bn = batchnorm(out_channels, use_bn, momentum=momentum, track=track)
        self.activation = activation

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x + skip)
        return self.activation(x)

    
class EmbeddingMod_out(nn.Module):
    """
    Embedding module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu):
        super(EmbeddingMod_out, self).__init__()
        pad = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size,
                         stride=(1,1,1), padding=pad)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))    
    


class EmbeddingMod(nn.Module):
    """
    Embedding module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu):
        super(EmbeddingMod, self).__init__()
        pad = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size,
                         stride=(1,2,2), padding=pad)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))

class OutputMod(nn.Module):
    """
    Embedding -> output module.

    Args:
        in_channels (int)
        out_spec (dictionary): Output specification.
        kernel_size (int or 3-tuple, optional)
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1,1,1)):
        super(OutputMod, self).__init__()
        padding = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size,
                        stride=1, padding=padding, bias=False)

    def forward(self, x):
        """
        Return an output list as "DataParallel" cannot handle an output
        dictionary.
        """
        return self.conv(x)