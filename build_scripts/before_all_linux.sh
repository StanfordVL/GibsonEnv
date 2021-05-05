# vi /var/tmp/test_script.sh
#!/bin/bash

# Upgrade the system
yum check-update

# Install dependency for mpi4pi
yum install openmpi-devel -y
export CC=/usr/lib64/openmpi/bin/mpicc

# Install cmake requirements
yum install -y freeglut* yum-utils glew* libXrandr* yum install libXinerama* install libXcursor* zeromq-devel boost boost-devel minizip* asciidoc

# Install cuda
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-rhel6.repo
yum clean all
yum install -y cuda

# Setup cuda
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
nvcc --version