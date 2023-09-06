import torch
import numpy as np
from scipy.stats import spearmanr

# a functionto compute dcg
def compute_dcg_at_k(relevances, k):
    dcg = 0
    for i in range(min(len(relevances), k)):
        dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
    return dcg

# a function for cosine similarity
def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    
    return torch.mm(a_norm, b_norm.transpose(0, 1))

# a function for dot product
def dotprod(a, b):
    """
    Computes the dotprod dotprod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dotprod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

# a function for euclidean distance
def euclidean(a, b):
    """
    Computes the euclidean dist(a[i], b[j]) + dotprod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dist(a[i], b[j]) + dotprod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b) 
        
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
        
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    return torch.cdist(a, b, p=2.0) + torch.mm(a, b.transpose(0, 1))

# a function for manhattan distance
def manhattan(a, b):
    """
    Computes the euclidean dist(a[i], b[j]) + dotprod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dist(a[i], b[j]) + dotprod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b) 
        
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
        
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    return torch.cdist(a, b, p=1.0) + torch.mm(a, b.transpose(0, 1))

# a function for chebyshev distance
def chebyshev(a, b):
    """
    Computes the euclidean dist(a[i], b[j]) + dotprod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dist(a[i], b[j]) + dotprod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
        
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b) 
        
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
        
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    
    return torch.cdist(a, b, p=float('inf'))