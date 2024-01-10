#!/bin/bash

#git submodule update --init --recursive --jobs 0


source $(conda info --base)/etc/profile.d/conda.sh

conda update -n base -c defaults conda

conda create -y -n adop python=3.9.7
conda activate adop

conda install -y ncurses=6.3 -c conda-forge
#conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.3 cudatoolkit=11.3 -c nvidia -c conda-forge
conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.4 cudatoolkit=11.4 -c nvidia -c conda-forge
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools=58.0.4 cmake=3.19.6 cffi typing_extensions future six requests dataclasses pybind11=2.6.2
conda install -y magma-cuda110 -c pytorch
conda install -y freeimage=3.17 jpeg=9d protobuf=3.13.0.1 -c conda-forge
