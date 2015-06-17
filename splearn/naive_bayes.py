# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from sklearn.base import copy
from sklearn.naive_bayes import (BaseDiscreteNB, BaseNB, BernoulliNB,
                                 GaussianNB, MultinomialNB)
from splearn.base import SparkClassifierMixin
from splearn.utils.validation import check_rdd


class SparkBaseNB(BaseNB, SparkClassifierMixin):

    """Abstract base class for distributed naive Bayes estimators"""

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def predict(self, X):
        """
        Perform classification on an RDD containing arrays of test vectors X.

        Parameters
        ----------
        X : RDD containing array-like items, shape = [m_samples, n_features]

        Returns
        -------
        C : RDD with arrays, shape = [n_samples]
            Predicted target values for X
        """
        check_rdd(X, (sp.spmatrix, np.ndarray))
        return X.map(
            lambda X: super(SparkBaseNB, self).predict(X))

    def predict_proba(self, X):
        """
        Return probability estimates for the RDD containing test vector X.

        Parameters
        ----------
        X : RDD containing array-like items, shape = [m_samples, n_features]

        Returns
        -------
        C : RDD with array-like items , shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the models for each RDD block. The columns correspond to the classes
            in sorted order, as they appear in the attribute `classes_`.
        """
        check_rdd(X, (sp.spmatrix, np.ndarray))
        return X.map(
            lambda X: super(SparkBaseNB, self).predict_proba(X))

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the RDD containing the
        test vector X.

        Parameters
        ----------
        X : RDD containing array-like items, shape = [m_samples, n_features]

        Returns
        -------
        C : RDD with array-like items, shape = [n_samples, n_classes]
            Returns the log-probability of the samples for each class in
            the model for each RDD block. The columns correspond to the classes
            in sorted order, as they appear in the attribute `classes_`.
        """
        check_rdd(X, (sp.spmatrix, np.ndarray))
        return X.map(
            lambda X: super(SparkBaseNB, self).predict_log_proba(X))


class SparkGaussianNB(GaussianNB, SparkBaseNB):

    """
    Distributed Gaussian Naive Bayes (SparkGaussianNB)

    Based on sklearn's GaussianNB, model averaging is based on the work of
    the Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

        http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.

    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class

    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class

    Examples
    --------
    TODO!
    """

    def fit(self, Z, classes=None):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        check_rdd(Z, {'X': (sp.spmatrix, np.ndarray), 'y': (sp.spmatrix, np.ndarray)})
        models = Z[:, ['X', 'y']].map(
            lambda X_y: self.partial_fit(X_y[0], X_y[1], classes))
        avg = models.sum()
        self.__dict__.update(avg.__dict__)
        return self

    def __add__(self, other):
        this = copy.deepcopy(self)
        this_class2idx = {cls: idx for idx, cls in enumerate(this.classes_)}
        other_class2idx = {cls: idx for idx, cls in enumerate(other.classes_)}
        for class_i in this.classes_:
            i = this_class2idx[class_i]
            j = other_class2idx[class_i]
            N_x = this.class_count_[i]
            N_y = other.class_count_[j]
            mu_x = this.theta_[i, :]
            mu_y = other.theta_[j, :]
            sigma_x = this.sigma_[i, :]
            sigma_y = this.sigma_[j, :]
            N_total = N_x + N_y

            mu_xy = N_x * mu_x + N_y * mu_y
            sigma_xy = (sigma_x * N_x + sigma_y * N_y +
                        (N_x * N_y * (mu_x - mu_y) ** 2) / N_total)

            this.theta_[i, :] = mu_xy / N_total
            this.sigma_[i, :] = sigma_xy / N_total
            this.class_count_[i] += N_y

        this.class_prior_[:] = this.class_count_ / np.sum(this.class_count_)
        return this


class SparkBaseDiscreteNB(BaseDiscreteNB, SparkBaseNB):

    """
    Abstract base class for distributed naive Bayes on discrete/categorical
    data. It provides the necessary methods for model averaging.

    """

    def __add__(self, other):
        """
        Add method for DiscreteNB models.

        Parameters
        ----------
        other : fitted splearn multinomilal NB model with class_count_
                and feature_count_ attribute
            Model to add.

        Returns
        -------
        model : splearn Naive Bayes model
            Model with updated coefficients.
        """
        # The rdd operator add does not consider __radd__ :(
        if other == 0:
            return self
        model = copy.deepcopy(self)
        model.class_count_ += other.class_count_
        model.feature_count_ += other.feature_count_
        model._update_class_log_prior()
        model._update_feature_log_prob()
        return model

    def fit(self, Z, classes=None):
        """
        TODO fulibacsi fix docstring
        Fit Multinomial Naive Bayes according to (X,y) pair
        which is zipped into TupleRDD Z.

        Parameters
        ----------
        Z : TupleRDD containing X [array-like, shape (m_samples, n_features)]
            and y [array-like, shape (m_samples,)] tuple
            Training vectors, where ,_samples is the number of samples in the
            block and n_features is the number of features, and y contains
            the target values.

        Returns
        -------
        self : object
            Returns self.
        """
        check_rdd(Z, {'X': (sp.spmatrix, np.ndarray), 'y': (sp.spmatrix, np.ndarray)})
        if 'w' in Z.columns:
            models = Z[:, ['X', 'y', 'w']].map(
                lambda X_y_w: self.partial_fit(
                    X_y_w[0], X_y_w[1], classes, X_y_w[2]
                )
            )
        else:
            models = Z[:, ['X', 'y']].map(
                lambda X_y: self.partial_fit(X_y[0], X_y[1], classes))
        avg = models.sum()
        self.__dict__.update(avg.__dict__)
        return self


class SparkMultinomialNB(MultinomialNB, SparkBaseDiscreteNB):
    pass


class SparkBernoulliNB(BernoulliNB, SparkBaseDiscreteNB):
    pass
