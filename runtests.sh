#!/usr/bin/env bash

# assumes run from python/test directory

if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

PYFORJ=`ls -1 $SPARK_HOME/python/lib/py4j-*-src.zip | head -1`

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:../
export PYTHONPATH=$PYTHONPATH:$PYFORJ

export PYTHONWARNINGS="ignore"

nosetests $@ --verbosity 2 --rednose --nologcapture

