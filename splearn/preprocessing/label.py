import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing.label import _check_numpy_unicode_bug
from sklearn.utils import column_or_1d

from ..base import SparkBroadcasterMixin, SparkTransformerMixin


class SparkLabelEncoder(LabelEncoder, SparkTransformerMixin,
                        SparkBroadcasterMixin):

    """Encode labels with value between 0 and n_classes-1.
    Read more in the :ref:`User Guide <preprocessing_targets>`.
    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.
    Examples
    --------
    `SparkLabelEncoder` can be used to normalize labels.
    >>> from splearn.preprocessing import SparkLabelEncoder
    >>> from splearn import BlockRDD
    >>>
    >>> data = ["paris", "paris", "tokyo", "amsterdam"]
    >>> y = BlockRDD(sc.parallelize(data))
    >>>
    >>> le = SparkLabelEncoder()
    >>> le.fit(y)
    >>> le.classes_
    array(['amsterdam', 'paris', 'tokyo'],
          dtype='|S9')
    >>>
    >>> test = ["tokyo", "tokyo", "paris"]
    >>> y_test = BlockRDD(sc.parallelize(test))
    >>>
    >>> le.transform(y_test).toarray()
    array([2, 2, 1])
    >>>
    >>> test = [2, 2, 1]
    >>> y_test = BlockRDD(sc.parallelize(test))
    >>>
    >>> le.inverse_transform(y_test).toarray()
    array(['tokyo', 'tokyo', 'paris'],
          dtype='|S9')
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
        mapper = super(SparkLabelEncoder, self).transform
        mapper = self.broadcast(mapper, y.context)
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
        mapper = super(SparkLabelEncoder, self).inverse_transform
        mapper = self.broadcast(mapper, y.context)
        return y.transform(mapper)
