#!/usr/bin/env bash

cd venv/src/tf-pose/tf_pose/pafprocess
swig -python -c++ pafprocess.i && python setup.py build_ext --inplace