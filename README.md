# scikit-learn on PySpark

Experimental repository for supporting scikit-learn functionality and API on PySpark.

# Run IPython from notebooks directory

```bash
PYTHONPATH=${PYTHONPATH}:.. IPYTHON_OPTS="notebook" ${SPARK_HOME}/bin/pyspark --master local\[4\] --driver-memory 2G
```
