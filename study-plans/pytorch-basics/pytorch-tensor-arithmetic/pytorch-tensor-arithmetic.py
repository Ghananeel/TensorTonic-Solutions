import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    if op == "add":
        return torch.add(x, y).tolist()
    elif op == "matmul":
        return torch.matmul(x,y).tolist()
    elif op == "multiply":
        return torch.mul(x,y).tolist()
    elif op == "power":
        return torch.pow(x, y).tolist()
    elif op == "max":
        return torch.max(x, y).tolist()