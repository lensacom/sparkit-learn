Sparkit-learn
=============

|Build Status| |PyPi| |Gitter|

**PySpark + Scikit-learn = Sparkit-learn**

GitHub: https://github.com/lensacom/sparkit-learn

About
=====

Sparkit-learn aims to provide scikit-learn functionality and API on
PySpark. The main goal of the library is to create an API that stays
close to sklearn's.

The driving principle was to *"Think locally, execute distributively."*
To accomodate this concept, the basic data block is always an array or a
(sparse) matrix and the operations are executed on block level.


Requirements
============

-  **Python 2.7.x or 3.4.x**
-  **Spark[>=1.3.0]**
-  NumPy[>=1.9.0]
-  SciPy[>=0.14.0]
-  Scikit-learn[>=0.16]



Run IPython from notebooks directory
====================================

.. code:: bash

    PYTHONPATH=${PYTHONPATH}:.. IPYTHON_OPTS="notebook" ${SPARK_HOME}/bin/pyspark --master local\[4\] --driver-memory 2G


Run tests with
==============

.. code:: bash

    ./runtests.sh


Quick start
===========

Sparkit-learn introduces three important distributed data format:

-  **ArrayRDD:**

   A *numpy.array* like distributed array

   .. code:: python

       from splearn.rdd import ArrayRDD

       data = range(20)
       # PySpark RDD with 2 partitions
       rdd = sc.parallelize(data, 2) # each partition with 10 elements
       # ArrayRDD
       # each partition will contain blocks with 5 elements
       X = ArrayRDD(rdd, bsize=5) # 4 blocks, 2 in each partition

   Basic operations:

   .. code:: python

       len(X) # 20 - number of elements in the whole dataset
       X.blocks # 4 - number of blocks
       X.shape # (20,) - the shape of the whole dataset

       X # returns an ArrayRDD
       # <class 'splearn.rdd.ArrayRDD'> from PythonRDD...

       X.dtype # returns the type of the blocks
       # numpy.ndarray

       X.collect() # get the dataset
       # [array([0, 1, 2, 3, 4]),
       #  array([5, 6, 7, 8, 9]),
       #  array([10, 11, 12, 13, 14]),
       #  array([15, 16, 17, 18, 19])]

       X[1].collect() # indexing
       # [array([5, 6, 7, 8, 9])]

       X[1] # also returns an ArrayRDD!

       X[1::2].collect() # slicing
       # [array([5, 6, 7, 8, 9]),
       #  array([15, 16, 17, 18, 19])]

       X[1::2] # returns an ArrayRDD as well

       X.tolist() # returns the dataset as a list
       # [0, 1, 2, ... 17, 18, 19]
       X.toarray() # returns the dataset as a numpy.array
       # array([ 0,  1,  2, ... 17, 18, 19])

       # pyspark.rdd operations will still work
       X.getNumPartitions() # 2 - number of partitions


- **SparseRDD:**

  The sparse counterpart of the *ArrayRDD*, the main difference is that the
  blocks are sparse matrices. The reason behind this split is to follow the
  distinction between *numpy.ndarray*s and *scipy.sparse* matrices.
  Usually the *SparseRDD* is created by *splearn*'s transformators, but one can
  instantiate too.

  .. code:: python

       # generate a SparseRDD from a text using SparkCountVectorizer
       from splearn.rdd import SparseRDD
       from sklearn.feature_extraction.tests.test_text import ALL_FOOD_DOCS
       ALL_FOOD_DOCS
       #(u'the pizza pizza beer copyright',
       # u'the pizza burger beer copyright',
       # u'the the pizza beer beer copyright',
       # u'the burger beer beer copyright',
       # u'the coke burger coke copyright',
       # u'the coke burger burger',
       # u'the salad celeri copyright',
       # u'the salad salad sparkling water copyright',
       # u'the the celeri celeri copyright',
       # u'the tomato tomato salad water',
       # u'the tomato salad water copyright')

       # ArrayRDD created from the raw data
       X = ArrayRDD(sc.parallelize(ALL_FOOD_DOCS, 4), 2)
       X.collect()
       # [array([u'the pizza pizza beer copyright',
       #         u'the pizza burger beer copyright'], dtype='<U31'),
       #  array([u'the the pizza beer beer copyright',
       #         u'the burger beer beer copyright'], dtype='<U33'),
       #  array([u'the coke burger coke copyright',
       #         u'the coke burger burger'], dtype='<U30'),
       #  array([u'the salad celeri copyright',
       #         u'the salad salad sparkling water copyright'], dtype='<U41'),
       #  array([u'the the celeri celeri copyright',
       #         u'the tomato tomato salad water'], dtype='<U31'),
       #  array([u'the tomato salad water copyright'], dtype='<U32')]

       # Feature extraction executed
       from splearn.feature_extraction.text import SparkCountVectorizer
       vect = SparkCountVectorizer()
       X = vect.fit_transform(X)
       # and we have a SparseRDD
       X
       # <class 'splearn.rdd.SparseRDD'> from PythonRDD...

       # it's type is the scipy.sparse's general parent
       X.dtype
       # scipy.sparse.base.spmatrix

       # slicing works just like in ArrayRDDs
       X[2:4].collect()
       # [<2x11 sparse matrix of type '<type 'numpy.int64'>'
       #   with 7 stored elements in Compressed Sparse Row format>,
       #  <2x11 sparse matrix of type '<type 'numpy.int64'>'
       #   with 9 stored elements in Compressed Sparse Row format>]

       # general mathematical operations are available
       X.sum(), X.mean(), X.max(), X.min()
       # (55, 0.45454545454545453, 2, 0)

       # even with axis parameters provided
       X.sum(axis=1)
       # matrix([[5],
       #         [5],
       #         [6],
       #         [5],
       #         [5],
       #         [4],
       #         [4],
       #         [6],
       #         [5],
       #         [5],
       #         [5]])

       # It can be transformed to dense ArrayRDD
       X.todense()
       # <class 'splearn.rdd.ArrayRDD'> from PythonRDD...
       X.todense().collect()
       # [array([[1, 0, 0, 0, 1, 2, 0, 0, 1, 0, 0],
       #         [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]]),
       #  array([[2, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0],
       #         [2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]]),
       #  array([[0, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0],
       #         [0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0]]),
       #  array([[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
       #         [0, 0, 0, 0, 1, 0, 2, 1, 1, 0, 1]]),
       #  array([[0, 0, 2, 0, 1, 0, 0, 0, 2, 0, 0],
       #         [0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1]]),
       #  array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1]])]

       # One can instantiate SparseRDD manually too:
       sparse = sc.parallelize(np.array([sp.eye(2).tocsr()]*20), 2)
       sparse = SparseRDD(sparse, bsize=5)
       sparse
       # <class 'splearn.rdd.SparseRDD'> from PythonRDD...

       sparse.collect()
       # [<10x2 sparse matrix of type '<type 'numpy.float64'>'
       #   with 10 stored elements in Compressed Sparse Row format>,
       #  <10x2 sparse matrix of type '<type 'numpy.float64'>'
       #   with 10 stored elements in Compressed Sparse Row format>,
       #  <10x2 sparse matrix of type '<type 'numpy.float64'>'
       #   with 10 stored elements in Compressed Sparse Row format>,
       #  <10x2 sparse matrix of type '<type 'numpy.float64'>'
       #   with 10 stored elements in Compressed Sparse Row format>]


-  **DictRDD:**

   A column based data format, each column with it's own type.

   .. code:: python

       from splearn.rdd import DictRDD

       X = range(20)
       y = list(range(2)) * 10
       # PySpark RDD with 2 partitions
       X_rdd = sc.parallelize(X, 2) # each partition with 10 elements
       y_rdd = sc.parallelize(y, 2) # each partition with 10 elements
       # DictRDD
       # each partition will contain blocks with 5 elements
       Z = DictRDD((X_rdd, y_rdd),
                   columns=('X', 'y'),
                   bsize=5,
                   dtype=[np.ndarray, np.ndarray]) # 4 blocks, 2/partition
       # if no dtype is provided, the type of the blocks will be determined
       # automatically

       # or:
       import numpy as np

       data = np.array([range(20), list(range(2))*10]).T
       rdd = sc.parallelize(data, 2)
       Z = DictRDD(rdd,
                   columns=('X', 'y'),
                   bsize=5,
                   dtype=[np.ndarray, np.ndarray])

   Basic operations:

   .. code:: python

       len(Z) # 8 - number of blocks
       Z.columns # returns ('X', 'y')
       Z.dtype # returns the types in correct order
       # [numpy.ndarray, numpy.ndarray]

       Z # returns a DictRDD
       #<class 'splearn.rdd.DictRDD'> from PythonRDD...

       Z.collect()
       # [(array([0, 1, 2, 3, 4]), array([0, 1, 0, 1, 0])),
       #  (array([5, 6, 7, 8, 9]), array([1, 0, 1, 0, 1])),
       #  (array([10, 11, 12, 13, 14]), array([0, 1, 0, 1, 0])),
       #  (array([15, 16, 17, 18, 19]), array([1, 0, 1, 0, 1]))]

       Z[:, 'y'] # column select - returns an ArrayRDD
       Z[:, 'y'].collect()
       # [array([0, 1, 0, 1, 0]),
       #  array([1, 0, 1, 0, 1]),
       #  array([0, 1, 0, 1, 0]),
       #  array([1, 0, 1, 0, 1])]

       Z[:-1, ['X', 'y']] # slicing - DictRDD
       Z[:-1, ['X', 'y']].collect()
       # [(array([0, 1, 2, 3, 4]), array([0, 1, 0, 1, 0])),
       #  (array([5, 6, 7, 8, 9]), array([1, 0, 1, 0, 1])),
       #  (array([10, 11, 12, 13, 14]), array([0, 1, 0, 1, 0]))]


Basic workflow
--------------

With the use of the described data structures, the basic workflow is
almost identical to sklearn's.

Distributed vectorizing of texts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SparkCountVectorizer
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from splearn.rdd import ArrayRDD
    from splearn.feature_extraction.text import SparkCountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    X = [...]  # list of texts
    X_rdd = ArrayRDD(sc.parallelize(X, 4))  # sc is SparkContext

    local = CountVectorizer()
    dist = SparkCountVectorizer()

    result_local = local.fit_transform(X)
    result_dist = dist.fit_transform(X_rdd)  # SparseRDD


SparkHashingVectorizer
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from splearn.rdd import ArrayRDD
    from splearn.feature_extraction.text import SparkHashingVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    X = [...]  # list of texts
    X_rdd = ArrayRDD(sc.parallelize(X, 4))  # sc is SparkContext

    local = HashingVectorizer()
    dist = SparkHashingVectorizer()

    result_local = local.fit_transform(X)
    result_dist = dist.fit_transform(X_rdd)  # SparseRDD


SparkTfidfTransformer
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from splearn.rdd import ArrayRDD
    from splearn.feature_extraction.text import SparkHashingVectorizer
    from splearn.feature_extraction.text import SparkTfidfTransformer
    from splearn.pipeline import SparkPipeline

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import Pipeline

    X = [...]  # list of texts
    X_rdd = ArrayRDD(sc.parallelize(X, 4))  # sc is SparkContext

    local_pipeline = Pipeline((
        ('vect', HashingVectorizer()),
        ('tfidf', TfidfTransformer())
    ))
    dist_pipeline = SparkPipeline((
        ('vect', SparkHashingVectorizer()),
        ('tfidf', SparkTfidfTransformer())
    ))

    result_local = local_pipeline.fit_transform(X)
    result_dist = dist_pipeline.fit_transform(X_rdd)  # SparseRDD


Distributed Classifiers
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from splearn.rdd import DictRDD
    from splearn.feature_extraction.text import SparkHashingVectorizer
    from splearn.feature_extraction.text import SparkTfidfTransformer
    from splearn.svm import SparkLinearSVC
    from splearn.pipeline import SparkPipeline

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline

    X = [...]  # list of texts
    y = [...]  # list of labels
    X_rdd = sc.parallelize(X, 4)
    y_rdd = sc.parralelize(y, 4)
    Z = DictRDD((X_rdd, y_rdd),
                columns=('X', 'y'),
                dtype=[np.ndarray, np.ndarray])

    local_pipeline = Pipeline((
        ('vect', HashingVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ))
    dist_pipeline = SparkPipeline((
        ('vect', SparkHashingVectorizer()),
        ('tfidf', SparkTfidfTransformer()),
        ('clf', SparkLinearSVC())
    ))

    local_pipeline.fit(X, y)
    dist_pipeline.fit(Z, clf__classes=np.unique(y))

    y_pred_local = local_pipeline.predict(X)
    y_pred_dist = dist_pipeline.predict(Z[:, 'X'])


Distributed Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from splearn.rdd import DictRDD
    from splearn.grid_search import SparkGridSearchCV
    from splearn.naive_bayes import SparkMultinomialNB

    from sklearn.grid_search import GridSearchCV
    from sklearn.naive_bayes import MultinomialNB

    X = [...]
    y = [...]
    X_rdd = sc.parallelize(X, 4)
    y_rdd = sc.parralelize(y, 4)
    Z = DictRDD((X_rdd, y_rdd),
                columns=('X', 'y'),
                dtype=[np.ndarray, np.ndarray])

    parameters = {'alpha': [0.1, 1, 10]}
    fit_params = {'classes': np.unique(y)}

    local_estimator = MultinomialNB()
    local_grid = GridSearchCV(estimator=local_estimator,
                              param_grid=parameters)

    estimator = SparkMultinomialNB()
    grid = SparkGridSearchCV(estimator=estimator,
                             param_grid=parameters,
                             fit_params=fit_params)

    local_grid.fit(X, y)
    grid.fit(Z)


Special thanks
==============

- scikit-learn community
- spylearn community
- pyspark community


.. |Build Status| image:: https://travis-ci.org/lensacom/sparkit-learn.png?branch=master
   :target: https://travis-ci.org/lensacom/sparkit-learn
.. |PyPi| image:: https://img.shields.io/pypi/v/sparkit-learn.svg
   :target: https://pypi.python.org/pypi/sparkit-learn
.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/lensacom/sparkit-learn
   :target: https://gitter.im/lensacom/sparkit-learn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

