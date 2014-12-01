# Sparkit-learn

[![Build Status](https://travis-ci.org/lensacom/sparkit-learn.png?branch=master)](https://travis-ci.org/lensacom/sparkit-learn)

__PySpark + Scikit-learn = Sparkit-learn__

# About

Sparkit-learn is an experimental repository for supporting scikit-learn
functionality and API on PySpark. The main goal of the library is to
create an API that stays close to sklearn's.

The driving principal was to _"Think locally, execute distributively."_ To
accomodate this conception, the basic data block is always an array or a
(sparse) matrix and the operations are executed on block level.

# Quick start

Sparkit-learn introduce two important distributed data format:

- __ArrayRDD:__

	A _numpy.array_ like distributed array

	```python
	from splearn.rdd import ArrayRDD

	data = range(20)
	# PySpark RDD with 2 partitions
	rdd = sc.parallelize(data, 2) # each partition with 10 elements
	# ArrayRDD
	# each partition will contain blocks with 5 elements
	X = ArrayRDD(rdd, block_size=5) # 4 blocks, 2 in each partition
	```
	Some basic operation:
	```python
	len(X) # 4 - number of blocks
	X.shape # (20,) - the shape of the whole dataset

	X # returns an ArrayRDD
	# <class 'splearn.rdd.ArrayRDD'> from PythonRDD...

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
	# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	X.toarray() # returns the dataset as a numpy.array
	# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

	# pyspark.rdd operations will still work
	X.numPartitions() # 2 - number of partitions
	```

- __DictRDD:__

	A column based data format, each row is a _numpy.array_.
	```python
	from splearn.rdd import DictRDD

	X = range(20)
	y = range(2) * 10
	# PySpark RDD with 2 partitions
	X_rdd = sc.parallelize(data_X, 2) # each partition with 10 elements
	y_rdd = sc.parallelize(data_y, 2) # each partition with 10 elements
	zipped_rdd = X_rdd.zip(y_rdd) # zip the two rdd's together
	# DictRDD
	# each partition will contain blocks with 5 elements
	Z = DictRDD(zipped_rdd, columns=('X', 'y'),  block_size=5) # 4 blocks, 2 in each partition

	# or:
	import numpy as np

	data = np.array([range(20), range(2)*10]).T
	rdd = sc.parallelize(data, 2)
	Z = DictRDD(rdd, columns=('X', 'y'),  block_size=5)
	```
	Basic operations:
	```python
	len(Z) # 4 - number of blocks
	Z.shape # (20,2) - the shape of the whole dataset

	Z # returns a DictRDD
	#<class 'splearn.rdd.DictRDD'> from PythonRDD...

	Z.collect()
	# [(array([0, 1, 2, 3, 4]), array([0, 1, 0, 1, 0])),
	#  (array([5, 6, 7, 8, 9]), array([1, 0, 1, 0, 1])),
	#  (array([10, 11, 12, 13, 14]), array([0, 1, 0, 1, 0])),
	#  (array([15, 16, 17, 18, 19]), array([1, 0, 1, 0, 1]))]

	Z[:, 'y'] # column select - returns a DictRDD
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
	```


The basic workflow is almost identical to sklearn's:

## Distributed vectorizing of texts

### SparkCountVectorizer

```python
from splearn.rdd import ArrayRDD
from splearn.feature_extraction.text import SparkCountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

X = [...]  # list of texts
X_rdd = ArrayRDD(sc.parallelize(X, 4))  # sc is SparkContext

local_vect = CountVectorizer()
dist_vect = SparkCountVectorizer()

result_local = local.fit_transform(X)
result_dist = dist.fit_transform(X_rdd)  # ArrayRDD
```

### SparkHashingVectorizer

```python
from splearn.rdd import ArrayRDD
from splearn.feature_extraction.text import SparkHashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

X = [...]  # list of texts
X_rdd = ArrayRDD(sc.parallelize(X, 4))  # sc is SparkContext

local_vect = HashingVectorizer()
dist_vect = SparkHashingVectorizer()

result_local = local.fit_transform(X)
result_dist = dist.fit_transform(X_rdd)  # ArrayRDD
```

### SparkTfidfTransformer

```python
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
dist_pipeline = Pipeline((
	('vect', SparkHashingVectorizer()),
	('tfidf', SparkTfidfTransformer())
))

result_local = local_pipeline.fit_transform(X)
result_dist = dist_pipeline.fit_transform(X_rdd)  # ArrayRDD
```

## Distributed Classifiers

```python
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
Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'))

local_pipeline = Pipeline((
	('vect', HashingVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', LinearSVC())
))
dist_pipeline = Pipeline((
	('vect', SparkHashingVectorizer()),
	('tfidf', SparkTfidfTransformer()),
	('clf', SparkLinearSVC())
))

local_pipeline.fit(X, y)
dist_pipeline.fit(Z, classes=np.unique(Z[:, 'y']))

y_pred_local = local_pipeline.predict(X)
y_pred_dist = dist_pipeline.predict(Z[:, 'X'])
```

## Distributed Model Selection

```python
from splearn.rdd import DictRDD
from splearn.grid_search import SparkGridSearchCV
from sklearn.naive_bayes import SparkMultinomialNB

from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

X = [...]
y = [...]
X_rdd = sc.parallelize(X, 4)
y_rdd = sc.parralelize(y, 4)
Z = DictRDD(X_rdd.zip(y_rdd), columns=('X', 'y'))

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
```

# Run IPython from notebooks directory

```bash
PYTHONPATH=${PYTHONPATH}:.. IPYTHON_OPTS="notebook" ${SPARK_HOME}/bin/pyspark --master local\[4\] --driver-memory 2G
```

# Requirements

- NumPy[>=1.9.0]
- SciPy[>=0.14.0]
- Scikit-learn[=0.16-git]
- Spark[>=1.1.0]

# Special thanks

We would like to thank to:
 - scikit-learn community
 - spylearn community
 - pyspark community
