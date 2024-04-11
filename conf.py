import torch

device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Using device: {device}', flush=True)
