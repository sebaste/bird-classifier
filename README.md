# bird-classifier

Identify bird species from images using a pre-trained machine learning model.

# Requirement(s)

* Python 3
* Pip

# Setup

```
pip install -r requirements.txt
```

# Running

```
python classifier.py
```

# Development

## Installation of Git pylint pre-commit hook

Run this from the root project directory:

```
$ [[ ! -e .git/hooks/pre-commit ]] && echo "#\!/bin/sh" > .git/hooks/pre-commit
$ echo "./classifier/tools/git-hooks/pre-commit-pylint" >> .git/hooks/pre-commit
$ chmod u+rx .git/hooks/pre-commit
```

# Troubleshooting

## Fixing Python certifications

If encountering SSL errors, run the `install-certs.py` script under tools/. This script is part of the brew package for OS X for instance, but is never run when installed.

## Fixing TensorFlow warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"

This source provides a solution to this problem: https://technofob.com/2019/06/14/how-to-compile-tensorflow-2-0-with-avx2-fma-instructions-on-mac/.

It involves downloading the TensorFlow repository and using bazel v0.26.1 build with the flag --copt=-mavx2 to compile TensorFlow with AVX2 support.
