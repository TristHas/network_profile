import numpy as np
import pandas as pd

from .computation_timing import t_profile_timings
from .computation_count import t_profile_theory
from .memory_profile import log_memory
from .helpers import DEFAULT_LTYPE

def _summarize_df(fwd_time, bwd_time, 
                 fw_flops, bw_flops, 
                 names, in_size, out_size):
    """
        Return a pandas.DataFrame summarizing the input sequences
    """
    return pd.DataFrame({"layer":names,
                         "fw_operation":fw_flops,
                         "forward_time":fwd_time,
                         "forward_effi": np.array(fw_flops)/np.array(fwd_time),
                         "bw_operation": bw_flops,
                         "backward_time":bwd_time,
                         "backward_effi":np.array(bw_flops)/np.array(bwd_time),
                         "bw_time/fw_time":np.array(bwd_time)/np.array(fwd_time),
                         "input_size":in_size,
                         "output_size":out_size
                    })

def t_profile_net(model, inp, layer_type=DEFAULT_LTYPE):
    """
    """
    fwd_time, bwd_time = t_profile_timings(model, inp)
    fw_flops, bw_flops,\
    names, in_size, out_size  = t_profile_theory(model, inp, layer_type)
    
    return _summarize_df(fwd_time, bwd_time, 
                         fw_flops, bw_flops, names, 
                         in_size, out_size)

