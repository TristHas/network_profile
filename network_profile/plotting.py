import matplotlib.pyplot as plt

def plt_time(data, mode="fwd", normalize=False, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    if mode == "fwd":
        x = data["forward_time (s)"]
    elif mode == "bwd":
        x = data["backward_time (s)"]
    else:
        x = data["backward_time (s)"] + data["forward_time (s)"]
    
    if normalize:
        title = "Time distribution"
        x  = 100 * x / x.sum()
        label = f"{mode} percent"
    else:
        title = "Time of execution"
        label = f"{mode} time"

    # Plotting the data
    x.plot(label=label, ax=ax)
    # Making the plot pretty
    format_ax(ax, data, title)    
    
def plt_MAC(data, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    label = "mac"
    title = "MAC"
    x = data["Mac"]
    # plot
    x.plot(label=label, ax=ax)
    # Making the plot pretty
    format_ax(ax, data, title)
      
def plt_FLOP(data, mode="fwd", normalize=False, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        
    if mode == "fwd":
        x = data["fw_operation (FLOP)"]
    elif mode == "bwd":
        x = data["bw_operation (FLOP)"]
    else:
        x = data["bw_operation (FLOP)"] + data["fw_operation (FLOP)"]
    
    if normalize:
        title = "FLOP distribution"
        x  = 100 * x / x.sum()
        label = f"{mode} percent"
    else:
        title = "FLOP count"
        label = f"{mode} flop"
        
    title = "FLOP"
    
    # plot
    x.plot(label=label, ax=ax)
    # Making the plot pretty
    format_ax(ax, data, title)

def plt_FLOP_seconds(data, mode="fwd", ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        
    if mode == "fwd":
        x = data["fw_operation (FLOP)"] / data["forward_time (s)"] 
    elif mode == "bwd":
        x = data["bw_operation (FLOP)"] / data["backward_time (s)"] 
    else:
        x = (data["bw_operation (FLOP)"] + data["fw_operation (FLOP)"]) /\
            (data["forward_time (s)"] + data["backward_time (s)"])
    
    title = "Computational efficiency (FLOP/s)"
    label = f"{mode} flop/s"
    # plot
    x.plot(label=label, ax=ax)
    # Making the plot pretty
    format_ax(ax, data, title)

    
def plt_arithmetic_intensity(data, mode="fwd", ax=None):
    """
        TODO: Fix MAC computations (bw and fw are different!)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        
    if mode == "fwd":
        x = data["fw_operation (FLOP)"] / data["Mac"] 
    elif mode == "bwd":
        x = data["bw_operation (FLOP)"] / data["Mac"] 
    else:
        x = (data["bw_operation (FLOP)"] + data["fw_operation (FLOP)"]) /\
            (data["Mac"] + data["Mac"])
    
    title = "Arithmetic intensity (FLOP/Mac)"
    label = f"{mode} flop/mac"
    # plot
    x.plot(label=label, ax=ax)
    # Making the plot pretty
    format_ax(ax, data, title)


def plt_memory(mem, rotation=90, ax=None):
    """
        TODO: Fix MAC computations (bw and fw are different!)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(30,10))

    index = range(len(mem["mem_all"]))
    label_vals = mem["layer_type"].values+" "+mem["hook_type"].values
    ax.plot(mem["mem_all"])
    ax.set_xticks(index)
    ax.set_xticklabels(labels=label_vals,  rotation=rotation)
    ax.xaxis.grid(True)
    
def format_ax(ax, data, title="", rotation=70):
    """
    """
    label_vals = data["layer"].values.tolist()
    index = data.index
    
    ax.set_xticks(index)
    ax.set_xticklabels(labels=label_vals,  rotation=rotation)
    ax.set_title(title)
    ax.legend()
    ax.xaxis.grid(True)

def plot_time_FLOPs_FLOP_MAC(data):

    fig, axes = plt.subplots(2, 2, figsize=(30,16))


    ax = axes[0,0]
    plt_time(data, mode="fwd", normalize=False, ax=ax)
    plt_time(data, mode="bwd", normalize=False, ax=ax)
    plt_time(data, mode="total", normalize=False, ax=ax)

    ax = axes[0,1]
    plt_FLOP_seconds(data, "fwd", ax=ax)
    plt_FLOP_seconds(data, "bwd", ax=ax)
    plt_FLOP_seconds(data, "total", ax=ax)

    ax = axes[1,0]
    plt_FLOP(data, mode="fwd", normalize=False, ax=ax)
    plt_FLOP(data, mode="bwd", normalize=False, ax=ax)
    plt_FLOP(data, mode="total", normalize=False, ax=ax)

    ax = axes[1,1]
    plt_MAC(data, ax)