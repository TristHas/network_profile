import torch
import pandas as pd

from .helpers import train_model, select_layers, DEFAULT_LTYPE

def get_gpu_mem(device=0):
    return torch.cuda.memory_allocated(device), torch.cuda.memory_cached(device)

def generate_mem_hook(handle_ref, mem, idx, hook_type, device):
    """
    """
    def hook(self, *args):
        """
        """
        if len(mem)==0:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"]+1
        
        mem_all, mem_cached = get_gpu_mem(device)
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })
    return hook

def add_memory_hooks(idx, mod, mem_log, hr, device=0):
    """
    """
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', device))
    hr.append(h)
    
    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', device))
    hr.append(h)
    
    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', device))
    hr.append(h)
    
def log_memory(model, inp):
    """
    """
    device = inp.device
    hr, mem_log = [],[]
    
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, hr, device)
        
    train_model(model, inp)    
    [h.remove for h in hr]
    return mem_log

def log_memory(model, inp, ltype=DEFAULT_LTYPE):
    """
    """
    # Init
    device = inp.device
    hr, mem_log = [],[]
    # Warm-up
    train_model(model, inp)
    # Profile
    for idx, module in enumerate(select_layers(model, ltype=ltype)):
        add_memory_hooks(idx, module, mem_log, hr, device)
    train_model(model, inp)  
    # Clean up
    [h.remove for h in hr]
    return pd.DataFrame(mem_log)

