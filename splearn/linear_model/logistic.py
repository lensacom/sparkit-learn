# encoding: utf-8

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .base import SparkLinearModelMixin
from ..utils.validation import check_rdd


class SparkLogisticRegression(LogisticRegression, SparkLinearModelMixin):

    """Distributed implementation of scikit-learn's Logistic classifier.

    Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr' and uses the
    cross-entropy loss, if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs' and
    'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    `liblinear` library, newton-cg and lbfgs solvers. It can handle both
    dense and sparse input. Use C-ordered arrays or CSR matrices containing
    64-bit floats for optimal performance; any other input format will be
    converted (and copied).

    The newton-cg and lbfgs solvers support only L2 regularization with primal
    formulation. The liblinear solver supports both L1 and L2 regularization,
    with a dual formulation only for the L2 penalty.

    Parameters
    ----------
    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The newton-cg and
        lbfgs solvers support only l2 penalties.
    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.
    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.
    intercept_scaling : float, default: 1
        Useful only if solver is liblinear.
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight : {dict, 'auto'}, optional
        Over-/undersamples the samples of each class according to the given
        weights. If not given, all classes are supposed to have weight one.
        The 'auto' mode selects weights inversely proportional to class
        frequencies in the training set.
    max_iter : int
        Useful only for the newton-cg and lbfgs solvers. Maximum number of
        iterations taken for the solvers to converge.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.
    solver : {'newton-cg', 'lbfgs', 'liblinear'}
        Algorithm to use in the optimization problem.
    tol : float, optional
        Tolerance for stopping criteria.
    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Works only for the 'lbfgs'
        solver.
    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        Coefficient of the features in the decision function.
    intercept_ : array, shape (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
    n_iter_ : int
        Maximum of the actual number of iterations across all classes.
        Valid only for the liblinear solver.

    References
    ----------
    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    """

    # TODO: REVISIT!

    # workaround to keep the classes parameter unchanged
    @property
    def classes_(self):
        return self._classes_

    @classes_.setter
    def classes_(self, value):
        pass

    def fit(self, Z, classes=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        Z : DictRDD containing (X, y) pairs
            X - Training vector
            y - Target labels
        classes : iterable
            The set of available classes

        Returns
        -------
        self : object
            Returns self.
        """
        check_rdd(Z, {'X': (sp.spmatrix, np.ndarray)})
        # possible improve to partial_fit in partisions and then average
        # in final reduce
        self._classes_ = np.unique(classes)
        return self._spark_fit(SparkLogisticRegression, Z)

    def predict(self, X):
        """Distributed method to predict class labels for samples in X.

        Parameters
        ----------
        X : ArrayRDD containing {array-like, sparse matrix}
            Samples.

        Returns
        -------
        C : ArrayRDD
            Predicted class label per sample.
        """
        check_rdd(X, (sp.spmatrix, np.ndarray))
        return self._spark_predict(SparkLogisticRegression, X)
