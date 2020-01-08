import tvm
from topi.nn.util import get_pad_tuple3d
from topi.util import simplify
from topi.nn.pad import pad
from topi.nn.util import get_const_int
from topi.util import get_const_tuple

def group_conv3d_nchw(Input, Filter, stride, padding, dilation, groups, out_dtype = None):  
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3
    if isinstance(stride, int):
        stride_z, stride_h = stride_w = stride
    else:
        stride_z, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_z, dilation_h = dilation_w = dilation
    else:
        dilation_z, dilation_h, dilation_w = dilation

    batch, in_channel, in_z, in_height, in_width = get_const_tuple(Input.shape)
    num_filter, _, kernel_z, kernel_h, kernel_w = get_const_tuple(Filter.shape)

    assert in_channel % groups == 0, "input channels must divide group size"
    assert num_filter % groups == 0, "output channels must divide group size"

    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (kernel_z, kernel_h, kernel_w))
    
    # compute the output shape
    out_channel = num_filter
    out_z = simplify(
        (in_z - (kernel_z - 1) * dilation_z - 1 + pad_front + pad_back) // stride_z + 1)
    out_height = simplify(
        (in_height - (kernel_h - 1) * dilation_h - 1 + pad_top + pad_down) // stride_h + 1)
    out_width = simplify(
        (in_width - (kernel_w - 1) * dilation_w - 1 + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_front, pad_top, pad_left]
    pad_after = [0, 0, pad_back, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = tvm.reduce_axis((0, in_channel // groups), name='rc')
    rz = tvm.reduce_axis((0, kernel_z), name='rz')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    return tvm.compute(
        (batch, out_channel, out_z, out_height, out_width),
        lambda nn, ff, zz, yy, xx: tvm.sum(
            temp[nn, ff // (num_filter//groups) * (in_channel//groups) + rc,
                 zz * stride_z + rz * dilation_z,
                 yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, rz, ry, rx].astype(out_dtype),
            axis=[rc, rz, ry, rx]), tag='group_conv3d_nchw')