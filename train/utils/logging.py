import time
import inspect
import functools
import torch.distributed as dist
from tqdm import tqdm
import wandb
from train.utils.comm import gather_and_concat_list

def progress_bar(*args, **kwargs):
    """
    Reference: RL2/utils/logging.py lines 9-16
    """
    return tqdm(
        *args,
        position=1,
        leave=False,
        disable=(dist.get_rank() != 0),
        **kwargs
    )

def time_logger(name):
    """
    Reference: RL2/utils/logging.py lines 18-34
    """
    def decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        assert "step" in param_names
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            step = kwargs.get("step", args[param_names.index("step")])
            start = time.time()
            output = func(*args, **kwargs)
            if dist.get_rank() == 0:
                wandb.log({
                    f"timing/{name}": time.time() - start
                }, step=step)
            return output
        return wrapper
    return decorator

def gather_and_log(metrics, device_mesh, step):
    """
    Reference: RL2/utils/logging.py lines 36-50
    """
    metrics = {
        k: gather_and_concat_list(v, device_mesh)
        for k, v in metrics.items()
    }
    if dist.get_rank() == 0:
        metrics = {
            k: sum(v) / (1.0 if k == "loss" else len(v))
            for k, v in metrics.items()
        }
        tqdm.write(f"Step {step}, " + ", ".join([
            f"{k}: {v:.3g}" for k, v in metrics.items()
        ]))
        wandb.log(metrics, step=step)

def gather_and_reduce(lst, device_mesh):
    """
    Reference: RL2/utils/logging.py lines 52-56
    """
    lst = gather_and_concat_list(lst, device_mesh)
    if dist.get_rank() == 0:
        return sum(lst)

def rank0_log(metrics, step):
    """
    Reference: RL2/utils/logging.py lines 58-70
    """
    if not dist.get_rank() == 0:
        return
    
    metrics = {
        k: sum(v) / len(v)
        for k, v in metrics.items()
    }
    tqdm.write(f"Step {step}, " + ", ".join([
        f"{k}: {v:.3g}" for k, v in metrics.items()
    ]))
    wandb.log(metrics, step=step)