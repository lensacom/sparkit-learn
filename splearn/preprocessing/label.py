from ..base import SparkTransformerMixin, SparkBroadcasterMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.preprocessing.label import _check_numpy_unicode_bug

import numpy as np

class SparkLabelEncoder(LabelEncoder, SparkTransformerMixin, SparkBroadcasterMixin):
    """Encode labels with value between 0 and n_classes-1.
    Read more in the :ref:`User Guide <preprocessing_targets>`.
    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.
    Examples
    --------
    `LabelEncoder` can be used to normalize labels.
    >>> from splearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])
    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
    """

    __transient__ = ['classes_']

    def fit(self, y):
        """Fit label encoder
        Parameters
        ----------
        y : ArrayRDD (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """

        def mapper(y):
            y = column_or_1d(y, warn=True)
            _check_numpy_unicode_bug(y)
            return np.unique(y)

        def reducer(a, b):
            return np.unique(np.concatenate((a, b)))

        self.classes_ = y.map(mapper).reduce(reducer)

        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels
        Parameters
        ----------
        y : ArrayRDD [n_samples]
            Target values.
        Returns
        -------
        y : ArrayRDD [n_samples]
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : ArrayRDD [n_samples]
            Target values.
        Returns
        -------
        y : ArrayRDD [n_samples]
        """
        mapper = self.broadcast(super(SparkLabelEncoder, self).transform, y.context)
        return y.transform(mapper)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        y : ArrayRDD [n_samples]
        """
        mapper = self.broadcast(super(SparkLabelEncoder, self).inverse_transform, y.context)
        return y.transform(mapper)