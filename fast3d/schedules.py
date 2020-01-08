import tvm
from tvm import autotvm
from topi.util import get_const_tuple
from topi.nn.util import get_pad_tuple3d
from topi.util import simplify
from topi.nn.pad import pad
from topi.nn.util import get_const_int
import topi

from .operations import group_conv3d_nchw
#from .util import *

@autotvm.template
def schedule_direct_3d_cuda(inshape, kershape, stride, padding, dilation):
    """
        schedule optimized for batch size=1
    """
    data = tvm.placeholder(inshape)
    kernel = tvm.placeholder(kershape)
    
    conv = topi.nn.conv3d_ncdhw(data, kernel, stride, padding, dilation)
    
    s = tvm.create_schedule(conv.op)
    cfg = autotvm.get_config()

    
    ##### space definition begin #####
    n, f, d, y, x = s[conv].op.axis
    rc, rd, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_d", d, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_rd", rd, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    # fallback support
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.target_name, target.model, 'conv3d', 'direct')
        cfg.fallback_with_reference_log(ref_log)
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
        s[kernel].compute_inline()

        
        
    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

        
        
    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    
    # tile and bind spatial axes
    n, f, d, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    bd, vd, td, di = cfg["tile_d"].apply(s, output, d)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].reorder(bf, bd, by, bx, vf, vd, vy, vx, tf, td, ty, tx, fi, di, yi, xi)

    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(s[output].fuse(bd, by), tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vd, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(s[output].fuse(td, tf), tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, d, y, x = s[OL].op.axis
    rc, rd, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    rdo, rdi = cfg['tile_rd'].apply(s, OL, rd)
    ryo, ryi = cfg['tile_ry'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, OL, rx)
    s[OL].reorder(rco, rdo, ryo, rxo, rci, rdi, ryi, rxi, n, f, d, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, d, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, d, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        td, fused = s[load].split(fused, nparts=cfg["tile_d"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(s[load].fuse(td, ty), tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    N, CO, OD, OH, OW = get_const_tuple(output.shape)
    _, KD, KH, KW, CI = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OD * OH * OW * CO * CI * KD * KH * KW)
    return  s, [data, kernel, output]

@autotvm.template
def schedule_conv3d_transpose_nchw_cuda(inshape, kershape, strides, padding, output_padding, out_dtype):
    """
        TOPI Schedule callback for conv3d transpose operator.
    """
    cfg = autotvm.get_config()
    Input = tvm.placeholder(inshape)
    Filter = tvm.placeholder(kershape)
    
    conv = topi.cuda.conv3d_transpose_nchw.conv3d_transpose_nchw_cuda(cfg, Input, Filter, strides, padding, output_padding, out_dtype)
    conv = [conv] if isinstance(conv, tvm.tensor.Tensor) else conv
    
    s = tvm.create_schedule([x.op for x in conv])
    conv = conv[0]
    ##### space definition begin #####
    n, f, d, y, x = s[conv].op.axis
    rc, rd, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_d", d, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_rd", ry, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    # fallback support
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.target_name, target.model, 'conv3d', 'direct')
        cfg.fallback_with_reference_log(ref_log)
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
        s[kernel].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile and bind spatial axes
    n, f, d, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    bd, vd, td, di = cfg["tile_d"].apply(s, output, d)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].reorder(bf, bd, by, bx, vf, vd, vy, vx, tf, td, ty, tx, fi, di, yi, xi)

    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(s[output].fuse(bd, by), tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vd, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(s[output].fuse(td, tf), tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, d, y, x = s[OL].op.axis
    rc, rd, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    rdo, rdi = cfg['tile_rd'].apply(s, OL, rd)
    ryo, ryi = cfg['tile_ry'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, OL, rx)
    s[OL].reorder(rco, rdo, ryo, rxo, rci, rdi, ryi, rxi, n, f, d, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, d, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, d, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        td, fused = s[load].split(fused, nparts=cfg["tile_d"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(s[load].fuse(td, ty), tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    N, CO, OD, OH, OW = get_const_tuple(output.shape)
    _, KD, KH, KW, CI = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OD * OH * OW * CO * CI * KD * KH * KW)

    return s, [Input, Filter, conv]


@autotvm.template  
def schedule_conv3d_nchw_cuda(inshape, grad_shape, stride, padding, dilation, groups, out_dtype=None):
    """
    schedule_conv3d_nchw_cuda
    """
    inp = tvm.placeholder(inshape)
    grad_r = tvm.placeholder(grad_shape)
    conv = group_conv3d_nchw(inp,grad_r,stride, padding, dilation, groups, out_dtype)
    
    s = tvm.create_schedule(conv.op)
    cfg = autotvm.get_config()
    num_filters = get_const_int(conv.shape[1])
    
    
    pad_data, kernel = s[conv].op.input_tensors
    s[pad_data].compute_inline()
    
    
    ##### space definition begin #####
    n, f, d, y, x = s[conv].op.axis
    rc, rd, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_n", n, num_outputs=4)
#     cfg.define_split("tile_g", cfg.axis(groups), num_outputs=2)
    cfg.define_split("tile_f", cfg.axis(num_filters // groups), num_outputs=4)
    cfg.define_split("tile_d", d, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_rd", rd, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.current_target()
    if target.target_name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope('local')
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile and bind spatial axes
    n, f, d, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

#     g, f = s[output].split(f, nparts=groups)
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
#     bg, vg = cfg["tile_g"].apply(s, output, g)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    bd, vd, td, di = cfg["tile_d"].apply(s, output, d)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bf, bd, by, bx, vn, vf, vd, vy, vx, tn, tf, td, ty, tx, ni, di, fi, yi, xi)
    s[output].bind(bn, tvm.thread_axis("blockIdx.z"))
#     s[output].bind(s[output].fuse(bg, bf), tvm.thread_axis("blockIdx.y"))
    s[output].bind(bf, tvm.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(bd, by, bx), tvm.thread_axis("blockIdx.x"))
    s[output].bind(vn, tvm.thread_axis("vthread"))
#     s[output].bind(vg, tvm.thread_axis("vthread"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vd, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))

    cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, tvm.thread_axis("threadIdx.z"))
        s[output].bind(tf, tvm.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(td, ty, tx)
        s[output].bind(tyx, tvm.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[output], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2] * cfg["tile_d"].size[2]
    else:
        s[output].bind(s[output].fuse(tn, tf, td), tvm.thread_axis("threadIdx.z"))
        s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[OL].compute_at(s[output], tx)

        # number of threads
        n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2] * cfg["tile_d"].size[2]
        n_ty = cfg["tile_y"].size[2]
        n_tx = cfg["tile_x"].size[2]

    # tile reduction axes
    n, f, d, y, x = s[OL].op.axis
    rc, rd, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, OL, rc)
    rdo, rdi = cfg['tile_rd'].apply(s, OL, rd)
    ryo, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxi = cfg['tile_ry'].apply(s, OL, rx)
    s[OL].reorder(rco, rdo, ryo, rxo, rci, rdi, ryi, rxi, n, f, d, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, d, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, d, y, x)
        fused, tx = s[load].split(fused, factor = n_tx)
        fused, ty = s[load].split(fused, factor = n_ty)
        fused, tz = s[load].split(fused, factor = n_tz)
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    N, CO, OZ, OH, OW = get_const_tuple(output.shape)
    _, CI_div_groups, KZ, KH, KW = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OZ * OH * OW * CO * CI_div_groups * KZ * KH * KW)
    return s, [inp, grad_r, output]