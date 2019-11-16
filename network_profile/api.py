import numpy as np
import pandas as pd
from si_prefix import si_format

from .computation_timing import t_profile_timings
from .computation_count import t_profile_theory
from .memory_profile import log_memory
from .helpers import DEFAULT_LTYPE

def _summarize_df(fwd_time, bwd_time, 
                 fw_flops, bw_flops, 
                 names, in_size, out_size, mac):
    """
        Return a pandas.DataFrame summarizing the input sequences
    """
    
    return pd.DataFrame({"layer":names,
                         "fw_operation (FLOP)":np.array(fw_flops),
                         "forward_time (s)":np.array(fwd_time) *1e-6,
                         "forward_effi (FLOPs)": (np.array(fw_flops)/(np.array(fwd_time)*1e-6)) ,
                         "bw_operation (FLOP)": np.array(bw_flops),
                         "backward_time (s)": np.array(bwd_time) *1e-6,
                         "backward_effi (FLOPs)":(np.array(bw_flops)/(np.array(bwd_time)*1e-6)) ,
                         "bw_time/fw_time":np.array(bwd_time)/np.array(fwd_time),
                         "input_size":in_size,
                         "output_size":out_size,
                         "Mac": mac
                    })

def _get_sci_precision_number(numbers):
    values = []
    for num in numbers:
        values.append(si_format(num,precision = 2))
    return values

def dataframe_readble(data):
    data_sci = data.copy()
    dataframe_columns = ("fw_operation (FLOP)","forward_time (s)","forward_effi (FLOPs)",
                         "bw_operation (FLOP)","backward_time (s)","backward_effi (FLOPs)",
                         "Mac")
    for col in dataframe_columns:
        data_sci[col] = _get_sci_precision_number(data[col])
    return data_sci

def t_profile_net(model, inp, layer_type=DEFAULT_LTYPE):
    """
    """
    fwd_time, bwd_time = t_profile_timings(model, inp)
    fw_flops, bw_flops,\
    names, in_size, out_size, mac  = t_profile_theory(model, inp, layer_type)
    
    return _summarize_df(fwd_time, bwd_time, 
                         fw_flops, bw_flops, names, 
                         in_size, out_size, mac)

