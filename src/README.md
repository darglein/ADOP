## Compile ADOP

 * Prepare Host System (Ubuntu)
```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt install g++-9
g++-9 --version # Should Print Version 9.4.0 or higher
```
 * Create Conda Environment
 
```shell
conda create -y -n adop python=3.8.1
conda activate adop

conda install -y cudnn=8.2.1.32 cudatoolkit-dev=11.1 -c nvidia -c conda-forge
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake=3.19.6 cffi typing_extensions future six requests dataclasses
conda install -y -c pytorch magma-cuda110
```

 * Compile Pytorch (Don't use the conda/pip package!)
 
 ```shell
conda activate adop
git clone git@github.com:pytorch/pytorch.git
cd pytorch
git checkout v1.9.1
git submodule update --init --recursive --jobs 0

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9
python setup.py install
 ```

 * Compile ADOP
 
```shell
conda activate adop
git clone git@github.com:darglein/ADOP.git
cd ADOP
git submodule update --init --recursive --jobs 0

export CONDA=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CC=gcc-9
export CXX=g++-9
export CUDAHOSTCXX=g++-9

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${CONDA}/lib/python3.8/site-packages/torch/;${CONDA}" ..
make -j10

```

### View Scenes in VR (with OpenVR+SteamVR)

 * Install SteamVR
 * Install OpenVR with conda:

```shell
conda install -c schrodinger openvr 
```
 * Rebuild Project
```shell
cd ADOP
rm -rf build

# Execute the steps again from
# Compile ADOP
```

 * Run a scene in the VR viewer

```shell
cd ADOP
./build/bin/VRviewer scenes/tt_playground Experiments/playground/ep400/
```****