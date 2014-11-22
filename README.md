# scikit-learn on PySpark

[![Build Status](https://travis-ci.org/lensacom/sparkit-learn.png?branch=master)](https://travis-ci.org/lensacom/sparkit-learn)

Experimental repository for supporting scikit-learn functionality and API on PySpark.

# Run IPython from notebooks directory

```bash
PYTHONPATH=${PYTHONPATH}:.. IPYTHON_OPTS="notebook" ${SPARK_HOME}/bin/pyspark --master local\[4\] --driver-memory 2G
```
