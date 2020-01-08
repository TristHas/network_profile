import tvm
from tvm import autotvm
import topi
#from util import *

peak = 0.0
count = 0
from matplotlib import pyplot as plt
from IPython.display import clear_output

def plot_callback():
    """
        Copied from tvm tutorial
        Plot the speed for each autotuning iteration
    """
    y = list()
    x = list()
    fig,ax = plt.subplots(1,1)
    def _callback(_, inputs, results):
        global peak
        global count
        for inp, res in zip(inputs, results):
            count += 1
            if res.error_no == 0:
                cost = np.mean(res.costs)
                perf = (inp.task.flop / 2) / cost  #auto tvm mut 2, + and *
#                 if perf > peak:
#                     print("reached new peak: {:.2f} GFLOPS".format(perf/1e9))
#                     peak = perf
                peak = perf
            x.append(count)
            y.append(peak)
            if count % 8 == 0:
                plt.plot(x, y)
                #plt.axhline(y = manual_flops, color='r', linestyle=':', label='manual baseline')
                plt.ylabel('performance (FLOP/S)')
                plt.xlabel('trials run')
                #plt.legend()
                clear_output()
                plt.show()
    return _callback

def auto_tune(func, args, log_name, n_trial, measure_option, target = 'cuda'):
    #auto tune
    task = autotvm.task.create(func, args = args, target = target)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial = n_trial,
               measure_option = measure_option,
               callbacks = [autotvm.callback.log_to_file(log_name), plot_callback(),],
               #callbacks = [autotvm.callback.log_to_file(log_name)]
              )
    
def time_of_best_config(func, x, k, y, ctx, number = 400):    
    evaluator = func.time_evaluator(func.entry_name, ctx = ctx, number = number)
    finish_time = evaluator(x, k, y).mean
    return finish_time

def the_best_config_model(log_name, func, para, env='cuda'):
    with autotvm.apply_history_best(log_name):
        with tvm.target.create(env):
            s, arg_bufs = func(*para)
            model = tvm.build(s, arg_bufs)
    return model


