#!/bin/bash

# Install dependency for mpi4pi
yum install openmpi3* openmpi* minizip* boost boost-devel SDL* libjpeg* portmidi* -y
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
make -j2
make install

# Install libglut
git clone https://github.com/funchal/libglu.git
cd libglu
./autogen.sh
make -j2
make install
cd..

# Install freeglut
wget https://github.com/dcnieho/FreeGLUT/archive/refs/tags/FG_3_2_1.zip
unzip FG_3_2_1.zip
cd FreeGLUT-FG_3_2_1/
cmake .
make
-j2
make install
cd ..

# Install zeromq
wget https://github.com/zeromq/libzmq/releases/download/v4.2.5/zeromq-4.2.5.zip
unzip zeromq-4.2.5.zip
cd zeromq-4.2.5
./configure
make -j2
make install
cd ..