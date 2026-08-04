import os
import subprocess

import torch


def select_device(prefer_idle=True):
    """
    Select a CUDA device without stealing GPU 0 by default.

    If CUDA_VISIBLE_DEVICES is already set by a launcher, respect it and use
    cuda:0 inside that visible namespace. Otherwise query nvidia-smi and choose
    the GPU with the lowest allocated memory.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        torch.cuda.set_device(0)
        return torch.device("cuda")

    if prefer_idle:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            candidates = []
            for line in result.stdout.strip().splitlines():
                if not line.strip():
                    continue
                idx, used = line.split(",")
                candidates.append((int(used.strip()), int(idx.strip())))
            if candidates:
                _, gpu_idx = min(candidates)
                torch.cuda.set_device(gpu_idx)
                return torch.device("cuda")
        except Exception:
            pass

    torch.cuda.set_device(0)
    return torch.device("cuda")


def describe_device(device):
    if device.type != "cuda":
        return str(device)
    idx = torch.cuda.current_device()
    return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
