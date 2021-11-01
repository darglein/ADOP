#!/bin/bash

git submodule update --init --recursive --jobs 0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate adop


cd External/pytorch



if command -v g++-9 &> /dev/null 
then
    export CC=$(which gcc-9)
    export CXX=$(which g++-9)
    export CUDAHOSTCXX=$(which g++-9)
    echo "Using g++-9"
elif command -v g++-7 &> /dev/null
then
    export CC=$(which gcc-7)
    export CXX=$(which g++-7)
    export CUDAHOSTCXX=$(which g++-7)
    echo "Using g++-7"
else
    echo "No suitable compiler found. Install g++-7 or g++-9"
    exit
fi


export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
