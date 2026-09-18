import torch

# Tiny tensor tests should not spawn a large OpenMP thread pool.
torch.set_num_threads(1)
