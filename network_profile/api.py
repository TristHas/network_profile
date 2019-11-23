import numpy as np
import pandas as pd
from si_prefix import si_format

from .computation_timing import t_profile_timings
from .computation_count import t_profile_theory
from .memory_profile import log_memory
from .helpers import DEFAULT_LTYPE
from .plotting import *

def _fix_timing_index(timings):
    mp = {"Conv":"Conv2d", "BatchNorm2d":"BatchNorm2d", "ReLU":"ReLU", 'MaxPool2d':'MaxPool2d'}
    idxes = timings.index.map(lambda x:x.split("_")[1])
    names = timings.index.map(lambda x:x.split("_")[0])
    names = names.map(lambda x:mp[x])
    layers = [f"{name}_{idx}" for name, idx in zip(names, idxes)]
    timings["layer"] = layers
    return timings
    
def _summarize_theory(data):
    fw_flops, bw_flops, names, in_sizes, out_sizes = data
    counter = {k:1 for k in np.unique(names)}
    new_names = []
    for name in names:
        new_names.append(f"{name}_{counter[name]}")
        counter[name] += 1

    df_theory = pd.DataFrame({"layer":new_names,
                     "fw_operation":fw_flops,
                     "bw_operation": bw_flops,
                     "input_size":in_sizes,
                     "output_size":out_sizes
                            })
    return df_theory

def _merge_time_theory(timings, theory):
    theory = _summarize_theory(theory)
    timings = _fix_timing_index(timings)
    return theory.merge(timings, on="layer")

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
    a = t_profile_timings(model, inp)
    b = t_profile_theory(model, inp, layer_type)
    c = _merge_time_theory(a, b)
    return c
