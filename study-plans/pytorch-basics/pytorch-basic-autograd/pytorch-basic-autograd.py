import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    x = torch.Tensor(values)
    x.requires_grad = True
    y = torch.sum(torch.add(2*x, torch.pow(x,3)))
    y.backward()

    return x.grad.tolist()