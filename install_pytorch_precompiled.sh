#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh

conda activate adop

CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

mkdir External/
cd External/


wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu116.zip -O  libtorch.zip
unzip libtorch.zip -d .


cp -rv libtorch/ $CONDA/lib/python3.9/site-packages/torch/