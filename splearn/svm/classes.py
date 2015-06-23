# encoding: utf-8

import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from splearn.linear_model.base import SparkLinearModelMixin

from ..utils.validation import check_rdd


class SparkLinearSVC(LinearSVC, SparkLinearModelMixin):

    """Distributed version of sklearn's Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better (to large numbers of
    samples).

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    loss : string, 'l1' or 'l2' (default='l2')
        Specifies the loss function. 'l1' is the hinge loss (standard SVM)
        while 'l2' is the squared hinge loss.
    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to `coef_`
        vectors that are sparse.
    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.
    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria
    multi_class: string, 'ovr' or 'crammer_singer' (default='ovr')
        Determines the multi-class strategy if `y` contains more than
        two classes.
        `ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`
        optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from an theoretical perspective
        as it is consistent it is seldom used in practice and rarely leads to
        better accuracy and is more expensive to compute.
        If `crammer_singer` is chosen, the options loss, penalty and dual will
        be ignored.
    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    intercept_scaling : float, optional (default=1)
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased
    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.
    verbose : int, (default=0)
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.
    max_iter : int, (default=1000)
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : array, shape = [n_features] if n_classes == 2 \
            else [n_classes, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.
        `coef_` is a readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.
    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    """

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
        self._classes_ = np.unique(classes)
        return self._spark_fit(SparkLinearSVC, Z)

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
        return self._spark_predict(SparkLinearSVC, X)
