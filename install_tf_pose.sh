#!/usr/bin/env bash

source venv/bin/activate
cd venv/src/tf-pose/tf_pose/pafprocess
swig -python -c++ pafprocess.i && python setup.py build_ext --inplace
