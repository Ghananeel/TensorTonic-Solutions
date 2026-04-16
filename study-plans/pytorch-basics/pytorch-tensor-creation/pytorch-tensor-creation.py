import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method == "ones":
        return torch.ones(shape).tolist()
    elif method == "zeros":
        return torch.zeros(shape).tolist()
    elif method == "full":
        tensor = torch.full((shape), value)
        return tensor.tolist()
