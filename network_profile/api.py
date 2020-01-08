import numpy as np
import pandas as pd
from si_prefix import si_format

from .computation_timing import t_profile_timings
from .computation_count import t_profile_theory
from .memory_profile import log_memory
from .helpers import DEFAULT_LTYPE
from .plotting import *

def _fix_timing_index(timings):
    mp = {"Conv":"Conv3d", "BatchNorm3d":"BatchNorm3d", "ReLU":"ReLU", 'MaxPool3d':'MaxPool3d'}
    idxes = timings.index.map(lambda x:x.split("_")[1])
    names = timings.index.map(lambda x:x.split("_")[0])
    names = names.map(lambda x:mp[x])
    layers = [f"{name}_{idx}" for name, idx in zip(names, idxes)]
    timings["layer"] = layers
    return timings
    
def _summarize_theory(data):
    fw_flops, bw_flops, names, in_sizes, out_sizes, mac = data
    counter = {k:0 for k in np.unique(names)}
    new_names = []
    for name in names:
        new_names.append(f"{name}_{counter[name]}")
        counter[name] += 1

    df_theory = pd.DataFrame({"layer":new_names,
                     "fw_operation":fw_flops,
                     "bw_operation": bw_flops,
                     "input_size":in_sizes,
                     "output_size":out_sizes,
                              "MAC":mac
                            })
    return df_theory

def _merge_time_theory(timings, theory):
    theory = _summarize_theory(theory)
    timings = _fix_timing_index(timings)
    return theory.merge(timings, on="layer", left_index = True)

def add_efficient(data):
    data["fwd_efficient"] =  data["fw_operation"] / data["fwd_node time"]
    data["bwd_efficient"] =  data["bw_operation"] / data["bwd_node time"]
    return data

def _get_sci_precision_number(numbers):
    values = []
    for num in numbers:
        values.append(si_format(num,precision = 2))
    return values

def dataframe_readble(data):
    data_sci = data.copy()
    dataframe_columns = ("fw_operation","fwd_node time","fwd_efficient",
                         "bw_operation","bwd_node time","bwd_efficient",
                         "MAC")
    for col in dataframe_columns:
        data_sci[col] = _get_sci_precision_number(data[col])
    return data_sci

def t_profile_net(model, inp, layer_type=DEFAULT_LTYPE):
    """
    """
    a = t_profile_timings(model, inp)
    b = t_profile_theory(model, inp, layer_type)
    c = _merge_time_theory(a, b)
    c = add_efficient(c)
    return c
