import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.Tensor(x)
    if method == "sigmoid":
        sigmoid = 1/(1 + torch.exp(-x))
        return sigmoid.tolist()
    elif method == "tanh":
        tanh = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        return tanh.tolist()
    elif method == "leaky_relu":
        upper = torch.maximum(x, torch.zeros_like(x))
        lower = torch.minimum(x, torch.zeros_like(x))
        lrelu = upper + 0.01 * lower
        return lrelu.tolist()
    else:
        relu = torch.maximum(x, torch.zeros_like(x))
        return relu.tolist()
        