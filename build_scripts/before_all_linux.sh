# vi /var/tmp/test_script.sh
#!/bin/bash

# Upgrade the system
yum check-update

# Upgrade cmake
yum install cmake wget
wget https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz
tar -zxvf cmake-3.5.2.tar.gz
cd cmake-3.5.2
./bootstrap
gmake
gmake install
cd ..

# Install dependency for mpi4pi
yum install openmpi* libmpi* -y
export CC=/usr/lib64/openmpi/bin/mpicc

# Install cmake requirements
yum install -y freeglut* yum-utils glew* libXrandr* yum install libXinerama* install libXcursor* boost boost-devel minizip* asciidoc zeromq3*

# Install zeromq
wget https://github.com/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.zip
unzip zeromq-4.2.5.zip
cd zeromq-4.2.5
./configure
make -j8
make install
cd ..

# Install cuda
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-rhel6.repo
yum clean all
yum install -y cuda

# Setup cuda
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
nvcc --version