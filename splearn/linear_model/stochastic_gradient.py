# encoding: utf-8

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier

from .base import SparkLinearModelMixin
from ..utils.validation import check_rdd


class SparkSGDClassifier(SGDClassifier, SparkLinearModelMixin):

    """Distributed version of sklearn's Linear classifiers
    (SVM, logistic regression, a.o.) with SGD training.

    Important! Due to distribution, the result on a local and a distributed
    dataset will differ.

    This estimator implements regularized linear models with stochastic
    gradient descent (SGD) learning: the gradient of the loss is estimated
    each sample at a time and the model is updated along the way with a
    decreasing strength schedule (aka learning rate). SGD allows minibatch
    (online/out-of-core) learning, see the partial_fit method.

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. The model it fits can be
    controlled with the loss parameter; by default, it fits a linear support
    vector machine (SVM).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    Parameters
    ----------
    loss : str, 'hinge', 'log', 'modified_huber', 'squared_hinge',\
                'perceptron', or a regression loss: 'squared_loss', 'huber',\
                'epsilon_insensitive', or 'squared_epsilon_insensitive'
        The loss function to be used. Defaults to 'hinge', which gives a
        linear SVM.
        The 'log' loss gives logistic regression, a probabilistic classifier.
        'modified_huber' is another smooth loss that brings tolerance to
        outliers as well as probability estimates.
        'squared_hinge' is like hinge but is quadratically penalized.
        'perceptron' is the linear loss used by the perceptron algorithm.
        The other losses are designed for regression but can be useful in
        classification as well; see SGDRegressor for a description.
    penalty : str, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'.
    alpha : float
        Constant that multiplies the regularization term. Defaults to 0.0001
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Defaults to 0.15.
    fit_intercept : bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.
    n_iter : int, optional
        The number of passes over the training data (aka epochs). The number
        of iterations is set to 1 if using partial_fit.
        Defaults to 5.
    shuffle : bool, optional
        Whether or not the training data should be shuffled after each epoch.
        Defaults to False.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.
    verbose : integer, optional
        The verbosity level
    epsilon : float
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.
    n_jobs : integer, optional
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation. -1 means 'all CPUs'. Defaults
        to 1.
    learning_rate : string, optional
        The learning rate:
        constant: eta = eta0
        optimal: eta = 1.0 / (t + t0) [default]
        invscaling: eta = eta0 / pow(t, power_t)
    eta0 : double
        The initial learning rate for the 'constant' or 'invscaling'
        schedules. The default value is 0.0 as eta0 is not used by the
        default schedule 'optimal'.
    power_t : double
        The exponent for inverse scaling learning rate [default 0.5].
    class_weight : dict, {class_label: weight} or "auto" or None, optional
        Preset for the class_weight fit parameter.
        Weights associated with classes. If not given, all classes
        are supposed to have weight one.
        The "auto" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies.
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    average : bool or int, optional
        When set to True, computes the averaged SGD weights and stores the
        result in the coef_ attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

    Attributes
    ----------
    coef_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,\
            n_features)
        Weights assigned to the features.
    intercept_ : array, shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.SGDClassifier()
    >>> clf.fit(X, Y)
    ... #doctest: +NORMALIZE_WHITESPACE
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
            learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
            penalty='l2', power_t=0.5, random_state=None, shuffle=False,
            verbose=0, warm_start=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    """

    def __init__(self, *args, **kwargs):
        super(SparkSGDClassifier, self).__init__(*args, **kwargs)
        self.average = True  # force averaging

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
        return self._spark_fit(SparkSGDClassifier, Z)

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
        return self._spark_predict(SparkSGDClassifier, X)
