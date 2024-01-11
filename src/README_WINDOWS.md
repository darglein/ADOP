# Compiling ADOP on Windows

Important: The Windows setup may not be working in all future commits, it was originally written for commit a433698.
It is also not tested as well as the Ubuntu setup, so prefer using that if issues arise.


#### Deprecated!

The Windows version is depricated, check out commit a433698 for the last working version.

### 1. Prerequisites

Supported Operating System
  * Windows 10

Required Software
  * Visual Studio 2019
  * CUDA 11.6 + cuDNN
  * powershell (included in Windows 10)
  * Anaconda3

Required GPU
  * Current Gen Nvidia GPU

Basic Setup
```shell
git clone git@github.com:darglein/ADOP.git
cd ADOP
git submodule update --init --recursive --jobs 0
```

### 2. Setup Environment

```shell
cd ADOP
conda create -y -n adop_windows python=3.9.7
conda activate adop_windows
conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.4 cudatoolkit=11.4 -c nvidia -c conda-forge
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake=3.19.6 cffi typing_extensions future six requests dataclasses pybind11=2.6.2
conda install -y freeimage=3.18 jpeg=9d protobuf=3.13 -c conda-forge

```

### 3. Install Pytorch from Source

 * We need a source build because the packaged pytorch was build using the old C++ ABI.


 ```shell
cd ADOP/External/pytorch

#start cmd
cmd

conda activate adop_windows
cd .jenkins/pytorch/win-test-helpers/installation-helpers

set USE_CUDA=1
#important: CUDA_VERSION=11.3, as the following install scripts don't work for 11.4
set CUDA_VERSION=11.3
set BUILD_TYPE=release
set TMP_DIR_WIN=%TMP%

install_magma.bat
install_mkl.bat
install_sccache.bat

cd ../../../..

set CUDA_PATH=%CONDA_PREFIX%\pkgs\cuda-toolkit\nvcc
set CMAKE_PREFIX_PATH=%CONDA_PREFIX%
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1

for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

python setup.py build --cmake --compiler=msvc

python setup.py install

## restart pc

```

### 4. Build ADOP

You may want to remove "ADOP/External/saiga/cmake/FindMKL.cmake" if MKL tools are not globally installed on your system, otherwise compiling may fail with <LNK1104 "MKL_LIBRARIES_CORE-NOTFOUND.lib" not found>.


```shell
cd ADOP
git submodule update --init --recursive --jobs 0

#start cmd
cmd
conda activate adop_windows

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%/Lib/site-packages/torch/;%CONDA_PREFIX%/Library;%CONDA_PREFIX%/Library/bin/;%CONDA_PREFIX%" -DCONDA_P_PATH="%CONDA_PREFIX%" ..

# start ADOP/build/ADOP.sln in VS2019
# select RelwithDebInfo as <Build_Config> and compile

# start from the command line with
# $./build/bin/<Build_Config>/adop_viewer.exe --scene_dir scenes/tt_playground
# or similar (see common README)

```

### Troubleshooting
  * cl.exe not found in PATH: Try restarting the PC or reset the Environment Variables set for the pytorch compiling
  * CMake: <LNK1104 "MKL_LIBRARIES_CORE-NOTFOUND.lib" not found> - See above and remove FindMKL.cmake and check your conda packages for completeness
