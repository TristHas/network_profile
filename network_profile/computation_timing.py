import pandas as pd
import numpy as np
from torch.autograd import profiler
from .helpers import train_model, select_layers, DEFAULT_LTYPE
from collections import OrderedDict

ltypes = {"Conv": {"fwd":{'conv2d','conv3d'}, "bwd":{'ThnnConv2DBackward', 'ThnnConvDepthwise2DBackward', "ThnnConvDepthwise2DBackward","CudnnConvolutionBackward", "SlowConvDilated3DBackward"}},
          "BatchNorm3d":{"fwd":'batch_norm',"bwd":{'NativeBatchNormBackward'}},
          "ReLU6":{"fwd":'hardtanh_',"bwd":{'HardtanhBackward1'}},
          "ReLU":{"fwd":'relu_',"bwd":{'ReluBackward1','ReluBackward0'}},
          "MaxPool2d":{"fwd":'max_pool2d',"bwd":{'MaxPool2DWithIndicesBackward'}},
          "MaxPool3d":{"fwd":'max_pool3d',"bwd":{'MaxPool3DWithIndicesBackward'}}, 
         }

layers = ltypes.keys()

def run_autograd_prof(func,*para):
    with profiler.profile(use_cuda = True) as prof:
        func_return = func(*para)
    return prof, func_return

def isin(cur, prev):
     return (cur[0] > prev[0]) and (cur[1] < prev[1])
    
def get_time_interval(event):
    return event.cpu_interval.start, event.cpu_interval.end

def get_time(x):
    """
    return the time for second
    """
    return x.cuda_time * 1e-6

get_name = lambda x: x.name


def identify_root_nodes(events):
    res = [False]

    previous_time_interval = get_time_interval(events[0])

    for event in events[1:]:
        current_time_interval = get_time_interval(event)
        if isin(current_time_interval, previous_time_interval):
            res.append(True)
        else:
            previous_time_interval = current_time_interval
            res.append(False)
    return res

def sort_by_root_nodes(events):
    roots = identify_root_nodes(events)
    result, idx = {}, 0
    for i, (event, child_node) in enumerate(list(zip(events, roots))):
        if child_node:
            result[idx]["children"].append(event)
        else:
            idx += 1
            result[idx] = {"parent":event, "children":[]}
    return result

def select(df, names): 
    """
    """
    if isinstance(names, str):
        names = {names}
    return df[df["name"].isin(names)]

def _map_fwd_bwd(df, ltype):
    fw = select(df, ltypes[ltype]["fwd"])
    bw = select(df, ltypes[ltype]["bwd"])
        
    mapping = pd.DataFrame({"fw_id":fw.idx.sort_values(ascending=True).values,
                            "bw_id": bw.idx.sort_values(ascending=False).values})
    
    mapping = {k:v for k,v in zip(
            fw.idx.sort_values(ascending=True).values,
            bw.idx.sort_values(ascending=False).values
    )}
    return mapping

def map_fwd_bwd(result, ltype_list):
    df = pd.DataFrame([(k, get_name(v["parent"])) \
                       for k,v in result.items()], columns=["idx", "name"])
    return {ltype:_map_fwd_bwd(df, ltype) for ltype in ltype_list}



def get_node_kernel(comp_node):
    """
        FIX ME
    """
    return comp_node['parent']

def handle_default(model):
    """
        Return all layer types in the model
    """
    return layers#NotImplementedError

def get_children_info(val):
    return {str(v.name):v.cuda_time for v in val['children']} 

def get_node_kernel(comp_node):
    """
        FIXME
    """
    return comp_node['parent']

def extract_node_info(comp_node):
    """
        Input  = comp_node
        Output = {  "node name":,
                    "node time":,
                    "kernel name":,
                    "kernel time":,
        }
    """
    info = { "node name"   : get_name(comp_node['parent']),
             "node time"   : get_time(comp_node['parent']),
             "kernel name" : get_name(get_node_kernel(comp_node)),
             "kernel time" : get_time(get_node_kernel(comp_node))
           }
    return info

def extract_nodes_info(comp_nodes):
    """
        Input  = comp_nodes
        Output = {"parent ker":xxx,"parent cuda time":xxx, "child":{xxx}}
    """
    return {idx: extract_node_info(comp_nodes[idx]) \
            for idx in comp_nodes}
    
def get_index(ltype, i):
    return f"{ltype}_{i}"

def merge_nodes(fwd_node, bwd_node):
    """
    """
    merged_node = {}
    for key in fwd_node.keys():
        merged_node[f"fwd_{key}"]=fwd_node[key]
    for key in bwd_node.keys():
        merged_node[f"bwd_{key}"]=bwd_node[key]    
    return merged_node
    
def merge_fwd_bwd_node(infos, fwd_bwd_map):
    """
        Input dict of infos, fwd_bwd_map
        
        Output:
            dict of order map, consist of fwd/bwd parent ker name, fwd/bwd cuda time
    """
    order_map = {}
    for ltype, fw_map_bw in fwd_bwd_map.items():
        # Make sure the idx of each module is sorted
        fw_keys = sorted(fw_map_bw.keys())
        for idx, (fw_key) in enumerate(fw_keys):
            #Get good nodes
            bw_key = fw_map_bw[fw_key]
            fw_node, bw_node = infos[fw_key], infos[bw_key]
            # Compute the value = merged node
            node = merge_nodes(fw_node, bw_node)
            # Compute index = ltype_idx
            idx = get_index(ltype, idx)
            order_map[idx] = node
    return order_map

def t_profile_timings(model, inp, layers=None):
    if layers is None:
        layers = handle_default(model)

    train_model(model, inp)
    cuda_r, _ = run_autograd_prof(train_model, model, inp)
    events = cuda_r.function_events

    comp_nodes = sort_by_root_nodes(events)
    fwd_bwd_map = map_fwd_bwd(comp_nodes, layers)

    infos = extract_nodes_info(comp_nodes)
    out = merge_fwd_bwd_node(infos, fwd_bwd_map)

    return pd.DataFrame(out).T