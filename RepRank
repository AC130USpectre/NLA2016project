import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.linalg.LinearOperator import matvec

def pplus(x):
    return x.clip(0)

def pminus(x):
    return x.clip(0)

def T(F, B, d, x):
    a1 = 0.7
    a2 = 0.7
    a3 = 0.7
    xp = pplus(x)
    xm = pminus(x)
    return a1*matvec(F,xp) + a2*matvec(B,xm) + a3*d

with load('../data/A.npz') as data:
    A = data['A']

F = normalize(A, axis=2, norm='l1')
B = normalize(A.T, axis=2, norm='l1')

for k in range(10):
    x = T(F,B,d,x)
