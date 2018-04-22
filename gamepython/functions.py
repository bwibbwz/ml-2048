import numpy as np

def relu(x):
    return max(0,x)

def to_basis(A):
    return np.log2(A)
