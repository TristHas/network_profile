from tvm.contrib.dlpack import to_pytorch_func

from .autotuning import the_best_config_model
from .schedules import schedule_direct_3d_cuda, schedule_conv3d_transpose_nchw_cuda, schedule_conv3d_nchw_cuda

def reshape_inp_weight_shape(x):
    bs,c,z,h,w = x
    return (bs*c,1,z,h,w)

def return_sch():
    return schedule_direct_3d_cuda, schedule_conv3d_transpose_nchw_cuda, schedule_conv3d_nchw_cuda

def best_functions(fw_log, bw_logs, inshape, kershape, outshape, stride, padding, dilation, groups, output_padding, bias):
    inp_sch, inp_g_sch, wei_g_sch = return_sch()
    
    func_fw = the_best_config_model(fw_log[0], inp_sch, (inshape, kershape, stride, padding, dilation))
    func_inp_g = the_best_config_model(bw_logs[0], inp_g_sch, (outshape, kershape, stride, padding, output_padding, 'float32'))
    func_wei_g = the_best_config_model(bw_logs[1], wei_g_sch, (tuple(reshape_inp_weight_shape(inshape)), tuple(reshape_inp_weight_shape(outshape)),
                                                               tuple([dilation,dilation,dilation]), tuple(padding), tuple(stride), groups))
    return func_fw, func_inp_g, func_wei_g

def best_pytorch_func(data_dict):
    func_dict = {"fw":[],"bw_inp_g":[],"bw_wei_g":[]}   

    for key, val in data_dict.items():
        fw_log, bw_logs, inshape, kershape, outshape, stride, padding, dilation, groups, output_padding, bias = val.values()

        func_fw, func_inp_g, func_wei_g = best_functions(fw_log, bw_logs, inshape, kershape, outshape, stride, padding, dilation, groups, output_padding, bias)

        func_dict['fw'].append(to_pytorch_func(func_fw))
        func_dict['bw_inp_g'].append(to_pytorch_func(func_inp_g))
        func_dict['bw_wei_g'].append(to_pytorch_func(func_wei_g))
    return func_dict

def func_map_keys(data_dict):
    keys = []
    for key,val in data_dict.items():
        fw_log, bw_log, inshape, kershape, outshape, stride, padding, dilation, groups, output_padding, bias = val.values()
        cin = kershape[1]
        cout = kershape[0]
        kernel = tuple(kershape[2:])
        keys.append((tuple(inshape), cin, cout, kernel, tuple(stride), tuple(padding), dilation, groups, bias))
        
    return keys

def get_func_map(data_dict):
    func_map = {}
    func_dict = best_pytorch_func(data_dict)
    keys = func_map_keys(data_dict)

    func_map = {}
    for key, fun_fw, fun_bw_inp, fun_bw_wei, in zip(keys, func_dict['fw'],func_dict['bw_inp_g'],func_dict['bw_wei_g']):
            func_map[key] =  {"fw":fun_fw, "bw_data":fun_bw_inp, "bw_weight": fun_bw_wei}
    return func_map