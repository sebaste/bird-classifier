#!/bin/bash

# Compile and install a customized TensorFlow.

BAZEL_VERSION=0.26.1

cd /tmp

# Install needed packages.
pip3 install -U --user pip six numpy wheel setuptools mock future>=0.17.1
pip3 install -U --user keras_applications==1.0.6 --no-deps
pip3 install -U --user keras_preprocessing==1.0.5 --no-deps

# Install bazel.
curl -fLO "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
chmod +x "bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh"
./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --user

# Clone the TensorFlow repository and checkout branch.
git clone https://github.com/tensorflow/tensorflow
git checkout r2.0

# Configure TensorFlow.
# Note: One can grep for 'environ_cp.get' in configure.py to see all the config variables one can set.
./configure

# Build using bazel.
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install TensorFlow.
pip3 install /tmp/tensorflow_pkg/tensorflow-2.0.0b1-cp37-cp37m-macosx_10_14_x86_64.whl 

unset BAZEL_VERSION
