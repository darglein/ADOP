#!/bin/bash

git submodule update --init --recursive --jobs 0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate adop



if command -v g++-9 &> /dev/null 
then
    export CC=gcc-9
    export CXX=g++-9
    export CUDAHOSTCXX=g++-9
    echo "Using g++-9"
elif command -v g++-7 &> /dev/null
then
    export CC=gcc-7
    export CXX=g++-7
    export CUDAHOSTCXX=g++-7
    echo "Using g++-7"
else
    echo "No suitable compiler found. Install g++-7 or g++-9"
    exit
fi

unset CUDA_HOME

mkdir build
cd build
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.9/site-packages/torch/;${CONDA}" ..
make -j10
