import io
import os
import subprocess
import torch
import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt

def get_gpu_mem(device=0):
    return torch.cuda.memory_allocated(device), torch.cuda.memory_cached(device)

def generate_mem_hook(handle_ref, mem, idx, hook_type, exp, device):
    """
    """
    def hook(self, *args):
        """
        """
        if len(mem)==0 or mem[-1]["exp"]!=exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"]+1
        
        mem_all, mem_cached = get_gpu_mem(device)
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })
    return hook

def add_memory_hooks(idx, mod, mem_log, exp, hr, device=0):
    """
    """
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', exp, device))
    hr.append(h)
    
    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', exp, device))
    hr.append(h)
    
    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', exp, device))
    hr.append(h)
    
def log_mem(model, inp, mem_log, exp, device=0):
    """
    """
    hr = []
    
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, exp, hr, device)
        
    out = model(inp)
    loss=out.sum()
    loss.backward()
    
    [h.remove for h in hr]
    
def pp(df, exp):
    """
    """
	df_exp = df[df.exp==exp]
	df_pprint =(
		df_exp.assign(
            open_layer = lambda ddf: ddf.hook_type.map(
                lambda x: {"pre":0, "fwd":1, "bwd":2}[x]).rolling(2).apply(lambda x: x[0]==0 and x[1] == 0
            )
        )
	   	.assign(
            close_layer = lambda ddf: ddf.hook_type.map(
                lambda x: {"pre":0, "fwd":1, "bwd":2}[x]).rolling(2).apply(lambda x: x[0]==1 and x[1] == 1)
        )
	   	.assign(indent_level =  lambda ddf: (ddf.open_layer.cumsum() - ddf.close_layer.cumsum()).fillna(0).map(int))
	   	.sort_values(by = "call_idx")
	   	.assign(mem_diff = lambda ddf: ddf.mem_all.diff()//2**20)
	)
	pprint_lines = [
        f"{'    '*row[1].indent_level}{row[1].layer_type} {row[1].hook_type}  {row[1].mem_diff or ''}"
        for row in  df_pprint.iterrows()
    ]
	for x in pprint_lines:
		print(x)
              
def print_code(x):
    	print(''.join(inspect.getsourcelines(x)[0]))
                                                
def plot_mem(
    df,
    exps=None,
    normalize_call_idx=True,
    normalize_mem_all=True,
    filter_fwd=False,
    return_df=False,
):
    if exps is None:
        exps=df.exp.drop_duplicates()
    
    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp==exp]
        
        if normalize_call_idx:
            df_.call_idx = df_.call_idx/df_.call_idx.max()
        
        
        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2**20
            
        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"]==layer_idx) & (df_["hook_type"]=="fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"]<=callidx_stop]
            #df_ = df_[df_.call_idx < df_[df_.layer_idx=='bwd'].call_idx.min()]
        
        df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)
    
    if return_df:
        return df_
