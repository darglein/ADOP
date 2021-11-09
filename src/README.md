# Compiling ADOP

### 1. Prerequisites

Supported Operating System
 * Ubuntu 18.04, 20.04

Supported Compiler 
 * g++-7
 * g++-9

Required Software
 * Anaconda

Basic Setup
```shell
git clone git@github.com:darglein/ADOP.git
cd ADOP
git submodule update --init --recursive --jobs 0
```

If you want the `adop_viewer` to work:
```shell
sudo apt install xorg-dev
```****

### 2. Setup Environment
 
```shell
cd ADOP
./create_environment.sh
```

### 3. Install Pytorch from Source
 
 * We need a source build because the packaged pytorch was build using the old C++ ABI. 
 
 ```shell
cd ADOP
./install_pytorch.sh
```

### 4. Build ADOP 

```shell
conda activate adop
git clone git@github.com:darglein/ADOP.git
cd ADOP
git submodule update --init --recursive --jobs 0

# Set this to either g++-7 or 9
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9

mkdir build
cd build
export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.9/site-packages/torch/;${CONDA}" ..
make -j10

```


### Building with VR support

 * Install Steam and SteamVR
 * Add openvr to the adop environment
```shell
conda activate adop
conda install -y -c schrodinger openvr 
```
 * Compile ADOP again

### Headless Build

 * On remote servers without Xorg we recommend the headless build.
 * Add the following flag to the `cmake` command of ADOP:
 * `-DADOP_HEADLESS=ON`
 * Note, `adop_viewer` can not be build headless.

### Troubleshooting

`libNVxxxx.so` not found when launching an executable
 * Add the `lib/` directory of the conda environment to `LD_LIBRARY_PATH`
 * Example: `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:~/anaconda3/envs/adop/lib`