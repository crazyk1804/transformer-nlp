import os
import sys
import torch

def get_device():
    if sys.platform == 'darwin':
        device_name = 'mps'
    else:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    return torch.device(device_name)