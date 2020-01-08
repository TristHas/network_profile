import matplotlib.pyplot as plt



FW_TIME = "fwd_node time"
BW_TIME = "bwd_node time"

FW_FLOP = "fw_operation"
BW_FLOP = "bw_operation"

MAC = "MAC"


def plt_time(data, mode="fwd", normalize=False, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    
    if mode == "fwd":
        x = data[FW_TIME]
    elif mode == "bwd":
        x = data[BW_TIME]
    else:
        x = data[BW_TIME] + data[FW_TIME]
    
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
    x = data[MAC]
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
        x = data[FW_FLOP]
    elif mode == "bwd":
        x = data[BW_FLOP]
    else:
        x = data[BW_FLOP] + data[FW_FLOP]
    
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
        x = data[FW_FLOP] / data[FW_TIME] 
    elif mode == "bwd":
        x = data[BW_FLOP] / data[BW_TIME] 
    else:
        x = (data[BW_FLOP] + data[FW_FLOP]) /\
            (data[FW_TIME] + data[BW_TIME])
    
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
        x = data[FW_FLOP] / data[MAC] 
    elif mode == "bwd":
        x = data[BW_FLOP] / data[MAC] 
    else:
        x = (data[BW_FLOP] + data[FW_FLOP]) /\
            (data[MAC] + data[MAC])
    
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
    index = range(len(index))####
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