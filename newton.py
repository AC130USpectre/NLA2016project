import numpy as np

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

a1 = 0.7
a2 = 0.7
a3 = 0.7

# we are looking for solution x == T(x)
def T(F, B, d, x):
    return a1 * F.dot(x.clip(0)) + a2 * B.dot(x.clip(-np.inf, 0)) + a3 * d

# Heaviside function
def h(x):
    ans = np.zeros_like(x)
    ans[x > 0.0] = 1.0
    return ans

# objective function
def obj(F, B, d, x):
    return np.linalg.norm(x - T(F, B, d, x)) ** 2 / 2

# Jacobian of residual
def Jac(F, B, d, x):
    return sp.sparse.eye(F.shape[0], format='csr') - a1 * F.multiply(sp.sparse.csc_matrix(h(x).reshape(-1, 1))) - a2 * B.multiply(sp.sparse.csc_matrix(h(-x).reshape(-1, 1)))

# derivative
def der(F, B, d, x):
    l = x - T(F, B, d, x)
    return Jac(F, B, d, x).T.dot(l)

# Hessian
def Hess(F, B, d, x):
    def matvec(x):
        return matvec.Jt.dot(matvec.J.dot(x))
    matvec.Jt = Jac(F, B, d, x).T.tocsr()
    matvec.J = Jac(F, B, d, x).tocsr()
    return sp.sparse.linalg.LinearOperator(F.shape, matvec)

# simple line search for Newton method
def LineSearch(F, B, d, x, dx):
    alpha = 1.0
    while alpha + 1.0 != 1.0:
        ans = x + alpha * dx
        if obj(F, B, d, ans) < obj(F, B, d, x):
            return ans
        else:
            alpha /= 2
    raise Exception("Machine precision achieved!")

# Newton method
def Newton(F, B, d, maxiter=100, x0=None, tol=1e-5, callback=None):
    if x0 is None:
        x_prev = d.copy()
    else:
        x_prev = x0.copy()

    for k in range(maxiter):
        dx, info = sp.sparse.linalg.cg(Hess(F, B, d, x_prev), -der(F, B, d, x_prev))
        x_next = LineSearch(F, B, d, x_prev, dx)
        if callback is not None:
            callback(k + 1, x_next)
        if np.linalg.norm(der(F, B, d, x_next)) < tol:
            break
        x_prev = x_next.copy()

    return (k + 1), x_next.copy()

# RepRank algorithm
def RepRank(F, B, d, maxiter=200, x0=None, tol=1e-5, callback=None):
    if x0 is None:
        x_prev = d.copy()
    else:
        x_prev = x0.copy()

    for k in range(maxiter):
        x_next = T(F, B, d, x_prev)
        n = np.linalg.norm(x_next - x_prev)
        if callback is not None:
            callback(x_next, n)
        if n < tol:
            break
        x_prev = x_next

    ans = x_next.copy()
    return(k + 1, ans)