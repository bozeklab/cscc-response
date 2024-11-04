import os
import fnmatch
import torch

def find_tar_files(directory='./'):
    tar_files = []
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, '*.tar'):
            tar_files.append(os.path.join(root, file))
    return tar_files

def move_tensors_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_tensors_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(move_tensors_to_cpu(v) for v in obj)
    elif isinstance(obj, list):
        return [move_tensors_to_cpu(v) for v in obj]
    else:
        return obj
