# vi /var/tmp/test_script.sh
#!/bin/bash

# Install dependency for mpi4pi
yum install openmpi* libmpi* libX* wget yum-utils boost boost-devel minizip* -y
export CC=/usr/lib64/openmpi/bin/mpicc

# Install pkg-config
wget https://pkg-config.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar xvfz pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2
./configure --with-internal-glib
make -j2
make install
cd ..

# Install glew
wget https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.zip
unzip glew-2.2.0.zip
cd glew-2.2.0
make
make install
cd ..


# Install libglut
git clone https://github.com/funchal/libglu.git
cd libglu
./autogen.sh
make install
cd ..

# Install freeglut
wget https://github.com/dcnieho/FreeGLUT/archive/refs/tags/FG_3_2_1.zip
unzip FG_3_2_1.zip
cd FreeGLUT-FG_3_2_1/
cmake .
make install
cd ..

# Install zeromq
wget https://github.com/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.zip
unzip zeromq-4.2.5.zip
cd zeromq-4.2.5
./configure
make
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