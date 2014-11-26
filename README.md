# scikit-learn on PySpark

[![Build Status](https://travis-ci.org/lensacom/sparkit-learn.png?branch=master)](https://travis-ci.org/lensacom/sparkit-learn)

Experimental repository for supporting scikit-learn functionality and API on PySpark.

# Run IPython from notebooks directory

```bash
PYTHONPATH=${PYTHONPATH}:.. IPYTHON_OPTS="notebook" ${SPARK_HOME}/bin/pyspark --master local\[4\] --driver-memory 2G
```

## ArrayRDD usage

```python
sc  # SparkContext

import numpy as np
from splearn.rdd import ArrayRDD

data = np.arange(16).reshape((8, 2))
rdd = sc.parallelize(data, 2)  # spark rdd
print data
print rdd.collect()

# wrapped rdd containing blocked numpy arrays
a4 = ArrayRDD(rdd)  # one block per partition
a2 = ArrayRDD(rdd, block_size=2)  # 8/2=4 blocks in two partitions

print a4.count(), a2.count()
print a4[0].first()
print a2[0].first()
print a2[-1:].collect()

print a2.tolist()
print a2.toarray()
```

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

## Distributed Classifier

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

