from operator import add

import numpy as np
import scipy.linalg as ln


def svd(blocked_rdd, k):
    """
    Calculate the SVD of a blocked RDD directly, returning only the leading k
    singular vectors. Assumes n rows and d columns, efficient when n >> d
    Must be able to fit d^2 within the memory of a single machine.
    Parameters
    ----------
    blocked_rdd : RDD
        RDD with data points in numpy array blocks
    k : Int
        Number of singular vectors to return
    Returns
    ----------
    u : RDD of blocks
        Left eigenvectors
    s : numpy array
        Singular values
    v : numpy array
        Right eigenvectors
    """

    # compute the covariance matrix (without mean subtraction)
    # TODO use one func for this (with mean subtraction as an option?)
    c = blocked_rdd.map(lambda x: (x.T.dot(x), x.shape[0]))
    prod, n = c.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))

    # do local eigendecomposition
    w, v = ln.eig(prod / n)
    w = np.real(w)
    v = np.real(v)
    inds = np.argsort(w)[::-1]
    s = np.sqrt(w[inds[0:k]]) * np.sqrt(n)
    v = v[:, inds[0:k]].T

    # project back into data, normalize by singular values
    u = blocked_rdd.map(lambda x: np.inner(x, v) / s)

    return u, s, v


def svd_em(blocked_rdd, k, maxiter=20, tol=1e-5, seed=None):
    """
    Calculate the SVD of a blocked RDD using an expectation maximization
    algorithm (from Roweis, NIPS, 1997) that avoids explicitly
    computing the covariance matrix, returning only the leading k
    singular vectors. Assumes n rows and d columns, does not require
    d^2 to fit into memory on a single machine.
    Parameters
    ----------
    blocked_rdd : RDD
        RDD with data points in numpy array blocks
    k : Int
        Number of singular vectors to return
    maxiter : Int, optional, default = 20
        Number of iterations to perform
    tol : Double, optional, default = 1e-5
        Tolerance for stopping iterative updates
    seed : Int, optional, default = None
        Seed for random number generator for initializing subspace
    Returns
    ----------
    u : RDD of blocks
        Left eigenvectors
    s : numpy array
        Singular values
    v : numpy array
        Right eigenvectors
    """

    n = blocked_rdd.shape[0]
    m = len(blocked_rdd.first()[0])

    def outerprod(x):
        return x.T.dot(x)

    if seed is not None:
        rng = np.random.RandomState(seed)
        c = rng.randn(k, m)
    else:
        c = np.random.randn(k, m)
    iter = 0
    error = 100

    # iteratively update subspace using expectation maximization
    # e-step: x = (cc')^-1 c y
    # m-step: c = y x' (xx')^-1
    while (iter < maxiter) & (error > tol):
        c_old = c
        # pre compute (cc')^-1 c
        c_inv = np.dot(c.T, ln.inv(np.dot(c, c.T)))
        premult1 = blocked_rdd._rdd.context.broadcast(c_inv)
        # compute (xx')^-1 through a map reduce
        xx = blocked_rdd.map(lambda x: outerprod(np.dot(x, premult1.value))) \
                        .reduce(add)
        xx_inv = ln.inv(xx)
        # pre compute (cc')^-1 c (xx')^-1
        premult2 = blocked_rdd._rdd.context.broadcast(np.dot(c_inv, xx_inv))
        # compute the new c through a map reduce
        c = blocked_rdd.map(lambda x: np.dot(x.T, np.dot(x, premult2.value))) \
                       .reduce(add)
        c = c.T

        error = np.sum((c - c_old) ** 2)
        iter += 1

    # project data into subspace spanned by columns of c
    # use standard eigendecomposition to recover an orthonormal basis
    c = ln.orth(c.T).T
    cov = blocked_rdd.map(lambda x: np.dot(x, c.T)) \
                     .map(lambda x: outerprod(x)) \
                     .reduce(add)
    w, v = ln.eig(cov / n)
    w = np.real(w)
    v = np.real(v)
    inds = np.argsort(w)[::-1]
    s = np.sqrt(w[inds[0:k]]) * np.sqrt(n)
    v = np.dot(v[:, inds[0:k]].T, c)
    u = blocked_rdd.map(lambda x: np.inner(x, v) / s)

    return u, s, v
