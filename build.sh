#!/bin/bash

## Core renderer
cd ./realenv/core/render/
wget https://www.dropbox.com/s/msd32wg144eew5r/coord.npy
bash build.sh
bash build_cuda.sh
python setup.py build_ext --inplace
cd -

## Core multi channel
wget https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip
unzip glfw-3.1.2.zip 
mv glfw-3.1.2 ./realenv/core/channels/external/glfw-3.1.2
make ./realenv/core/channels/build
cd ./realenv/core/channels/build
cmake .. & make -j 10
cd -

## Data set
cd ./realenv/data
mkdir dataset
wget https://www.dropbox.com/s/gtg09zm5mwnvro8/11HB6XZSh1Q.zip
unzip 11HB6XZSh1Q.zip
mv 11HB6XZSh1Q dataset
cd -

## Physics Models
cd ./realenv/core/physics
wget https://www.dropbox.com/s/vb3pv4igllr39pi/models.zip
unzip models.zip
cd -