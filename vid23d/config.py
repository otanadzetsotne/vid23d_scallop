import torch


# Select best available device
DEVICE = 'cpu'
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
