import torch


def pytest_sessionstart(session):
    # Tiny smoke tensors are faster and more stable without large CPU thread pools.
    torch.set_num_threads(1)
