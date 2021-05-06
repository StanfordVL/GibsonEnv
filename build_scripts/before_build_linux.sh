# Setup CUDA
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CC=/usr/lib64/openmpi/bin/mpicc

yum install -y wget
wget https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz
tar -zxvf cmake-3.5.2.tar.gz
cd cmake-3.5.2
./bootstrap
gmake
gmake install
cd ..

# Install pip requirements
pip install -r build_scripts/requirements.txt