#!/bin/bash

CONDA_PATH=~/anaconda3/

if test -f "$CONDA_PATH/etc/profile.d/conda.sh"; then
    echo "Found Conda at $CONDA_PATH"
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda --version
else
    echo "Could not find conda!"
fi


conda activate adop


#cd External/pytorch

mkdir External/
cd External/


wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.1%2Bcu113.zip -O  libtorch.zip
unzip libtorch.zip -d .


cp -rv libtorch/ $CONDA_PATH/envs/adop/lib/python3.9/site-packages/torch/




