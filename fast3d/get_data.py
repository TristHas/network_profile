
import tvm
import numpy as np

def get_rand(shape, devtype="cuda", devid=0):
    x = np.random.randn(*shape).astype('float32')
    ctx = tvm.context(devtype, devid)
    return tvm.nd.array(x, ctx)

def get_zero(shape, devtype="cuda", devid=0):
    x = np.zeros(shape).astype('float32')
    ctx = tvm.context(devtype, devid)
    return tvm.nd.array(x, ctx)

def get_ones(shape, devtype="cuda", devid=0):
    x = np.ones(shape).astype('float32')
    ctx = tvm.context(devtype, devid)
    return tvm.nd.array(x, ctx)

def get_input_kernel_output_shape(n,bs,c,hw,kz,kh,kw):
    cin = cout = int(c*((n)*2))
    H = W = Z = int(hw/(2**n))
    
    inshape  = (bs, cin, Z, H, W)
    kershape = (cout, cin, kz, kh, kw)
    outshape = (bs, cout, Z, H, W)
    return inshape, kershape, outshape

def cal_Conv3d(*para,groups=1):
    output_shape = para[1]
    HWZ = np.prod(output_shape[2:])
    bs = output_shape[0]
    kernel = np.prod(para[2])
    flop = kernel * HWZ * bs
    return flop/groups

def out_HWZ(x,k,s,p,d):
    return np.floor((x + 2 * p - d * (k - 1) - 1) / s + 1)

def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size):
    input_size = list(input_size)
    k = grad_output.dim() - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})"
                         .format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] +
                kernel_size[d])

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output of {})").format(
                     input_size, min_sizes, max_sizes,
                     grad_output.size()[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))
