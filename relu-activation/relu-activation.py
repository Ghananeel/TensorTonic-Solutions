import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    return np.clip(np.asarray(x), 0, None)