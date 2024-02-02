# Compiling ADOP on Windows

Important: The Windows setup may not be working in all future commits, it was originally written for commit a433698 and updated based on the setup used in [TRIPS](https://github.com/lfranke/TRIPS).
It is also not tested as well as the Ubuntu setup, so prefer using that if issues arise.

## Install Instructions Windows

### Software Requirements:

* VS2022
* CUDA 11.8
* Cudnn (copy into 11.8 folder as per install instructions) (we used version 8.9.7)
* conda (we used Anaconda3)

    [Start VS2022 once for CUDA integration setup]

### Clone Repo
```
git clone git@github.com:darglein/ADOP.git
cd ADOP/
git submodule update --init --recursive --jobs 0
```

### Setup Environment

```shell
conda update -n base -c defaults conda

conda create -y -n adop python=3.9.7

conda activate adop

conda install -y cmake=3.26.4
conda install -y -c intel mkl=2024.0.0
conda install -y -c intel mkl-static=2024.0.0
conda install openmp=8.0.1 -c conda-forge
```

### Install libtorch:


* Download: https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-1.13.1%2Bcu116.zip
* Unzip
* Copy into ADOP/External

Folder structure should look like:
```shell
ADOP/
    External/
        libtorch/
            bin/
            cmake/
            include/
            lib/
            ...
        saiga/
        ...
    src/
    ...
```

### Compile

```shell
cmake -Bbuild -DCMAKE_CUDA_COMPILER="$ENV:CUDA_PATH\bin\nvcc.exe" -DCMAKE_PREFIX_PATH=".\External\libtorch" -DCONDA_P_PATH="$ENV:CONDA_PREFIX" -DCUDA_P_PATH="$ENV:CUDA_PATH" -DCMAKE_BUILD_TYPE=RelWithDebInfo .
```
```shell
cmake --build build --config RelWithDebInfo -j
```

## Run Instructions Windows

Executable Paths on Windows need the build version added in the run path. You can start the viewer with:

```shell
./build/bin/RelWithDebInfo/adop_viewer.exe  --scene_dir scenes/tt_train
```

and the training with:
```shell
./build/bin/RelWithDebInfo/adop_train.exe --config configs/train_boat.ini
```
