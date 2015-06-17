from operator import add

import numpy as np
import scipy.linalg as ln
import scipy.sparse as sp
# from pyspark import AccumulatorParam
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import safe_sparse_dot

from ..base import SparkBroadcasterMixin
from ..rdd import DictRDD
from ..utils.validation import check_rdd


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


def svd_em(blocked_rdd, k, maxiter=20, tol=1e-6, compute_u=True, seed=None):
    """
    Calculate the SVD of a blocked RDD using an expectation maximization
    algorithm (from Roweis, NIPS, 1997) that avoids explicitly
    computing the covariance matrix, returning only the leading k
    singular vectors. Assumes n rows and d columns, does not require
    d^2 to fit into memory on a single machine.
    Parameters
    ----------
    blocked_rdd : ArrayRDD
        ArrayRDD with data points in numpy array blocks
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

    n, m = blocked_rdd.shape[:2]
    sc = blocked_rdd._rdd.context

    def outerprod(x):
        return x.T.dot(x)

    # global run_sum

    # def accumsum(x):
    #     global run_sum
    #     run_sum += x

    # class MatrixAccum(AccumulatorParam):

    #     def zero(self, value):
    #         return np.zeros(np.shape(value))

    #     def addInPlace(self, val1, val2):
    #         val1 += val2
    #         return val1

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
        premult1 = sc.broadcast(c_inv)

        # compute (xx')^-1 through a map reduce
        xx = blocked_rdd.map(lambda x: outerprod(safe_sparse_dot(x, premult1.value))) \
                        .treeReduce(add)

        # compute (xx')^-1 using an accumulator
        # run_sum = sc.accumulator(np.zeros((k, k)), MatrixAccum())
        # blocked_rdd.map(lambda x: outerprod(safe_sparse_dot(x, premult1.value))) \
        #            .foreachPartition(lambda l: accumsum(sum(l)))
        # xx = run_sum.value
        xx_inv = ln.inv(xx)

        # pre compute (cc')^-1 c (xx')^-1
        premult2 = blocked_rdd.context.broadcast(np.dot(c_inv, xx_inv))

        # compute the new c through a map reduce
        c = blocked_rdd.map(lambda x: safe_sparse_dot(x.T, safe_sparse_dot(x, premult2.value))) \
                       .treeReduce(add)

        # compute the new c using an accumulator
        # run_sum = sc.accumulator(np.zeros((m, k)), MatrixAccum())
        # blocked_rdd.map(lambda x: safe_sparse_dot(x.T, safe_sparse_dot(x, premult2.value))) \
        #            .foreachPartition(lambda l: accumsum(sum(l)))
        # c = run_sum.value
        c = c.T

        error = np.sum((c - c_old) ** 2)
        iter += 1

    # project data into subspace spanned by columns of c
    # use standard eigendecomposition to recover an orthonormal basis
    c = ln.orth(c.T).T
    cov = blocked_rdd.map(lambda x: safe_sparse_dot(x, c.T)) \
                     .map(lambda x: outerprod(x)) \
                     .treeReduce(add)
    w, v = ln.eig(cov / n)
    w = np.real(w)
    v = np.real(v)
    inds = np.argsort(w)[::-1]
    s = np.sqrt(w[inds[0:k]]) * np.sqrt(n)
    v = np.dot(v[:, inds[0:k]].T, c)
    if compute_u:
        v_broadcasted = blocked_rdd.context.broadcast(v)
        u = blocked_rdd.map(
            lambda x: safe_sparse_dot(x, v_broadcasted.value.T) / s)
        return u, s, v
    else:
        return s, v


class SparkTruncatedSVD(TruncatedSVD, SparkBroadcasterMixin):

    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). It is very similar to PCA,
    but operates on sample vectors directly, instead of on a covariance matrix.
    This means it can work with scipy.sparse matrices efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in sklearn.feature_extraction.text. In that
    context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithm: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on (X * X.T) or
    (X.T * X), whichever is more efficient.

    Parameters
    ----------
    n_components : int, default = 2
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : string, default = "randomized"
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, optional
        Number of iterations for randomized SVD solver. Not used by ARPACK.

    random_state : int or RandomState, optional
        (Seed for) pseudo-random number generator. If not given, the
        numpy.random singleton is used.

    tol : float, optional
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

    explained_variance_ : array, [n_components]
        The variance of the training samples transformed by a projection to
        each component.

    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from sklearn.random_projection import sparse_random_matrix
    >>> X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
    >>> svd = TruncatedSVD(n_components=5, random_state=42)
    >>> svd.fit(X) # doctest: +NORMALIZE_WHITESPACE
    TruncatedSVD(algorithm='randomized', n_components=5, n_iter=5,
            random_state=42, tol=0.0)
    >>> print(svd.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.07825... 0.05528... 0.05445... 0.04997... 0.04134...]
    >>> print(svd.explained_variance_ratio_.sum()) # doctest: +ELLIPSIS
    0.27930...

    See also
    --------
    PCA
    RandomizedPCA

    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061

    Notes
    -----
    SVD suffers from a problem called "sign indeterminancy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    """

    __transient__ = ['components_']

    def __init__(self, n_components=2, algorithm="em", n_iter=30,
                 random_state=None, tol=1e-7):
        super(SparkTruncatedSVD, self).__init__(
            n_components=n_components, algorithm=algorithm,
            n_iter=n_iter, random_state=random_state, tol=tol)

    def fit(self, Z):
        """Fit LSI model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(Z)
        return self

    def fit_transform(self, Z):
        """Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (sp.spmatrix, np.ndarray))
        if self.algorithm == "em":
            X = X.persist()  # boosting iterative svm
            Sigma, V = svd_em(X, k=self.n_components, maxiter=self.n_iter,
                              tol=self.tol, compute_u=False,
                              seed=self.random_state)
            self.components_ = V
            X.unpersist()
            return self.transform(Z)
        else:
            # TODO: raise warning non distributed
            return super(SparkTruncatedSVD, self).fit_transform(X.tosparse())

    def transform(self, Z):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (sp.spmatrix, np.ndarray))

        mapper = self.broadcast(
            super(SparkTruncatedSVD, self).transform, Z.context)
        return Z.transform(mapper, column='X', dtype=np.ndarray)

    def inverse_transform(self, Z):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        X = Z[:, 'X'] if isinstance(Z, DictRDD) else Z
        check_rdd(X, (sp.spmatrix, np.ndarray))

        mapper = self.broadcast(
            super(SparkTruncatedSVD, self).inverse_transform, Z.context)
        return Z.transform(mapper, column='X', dtype=np.ndarray)
