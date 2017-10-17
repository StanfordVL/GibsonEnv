#!/bin/bash

set -e

verify_cuda() {
	export CUDA_HOME=/usr/local/cuda-8.0
	export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 

	PATH=${CUDA_HOME}/bin:${PATH} 
	export PATH

	cuda-install-samples-8.0.sh ~/
	cd ~/NVIDIA_CUDA-8.0_Samples/1\_Utilities/deviceQuery  
	make --quiet
	./deviceQuery  | grep "Result = PASS" &
	greprc=$?
	if [[ $greprc -eq 0 ]] ; then
	    echo "Cuda Samples installed and GPU found"
	    echo "you can also check usage and temperature of gpus with nvidia-smi"
	else
	    if [[ $greprc -eq 1 ]] ; then
	        echo "Cuda Samples not installed, exiting..."
	        exit 1
	    else
	        echo "Some sort of error, exiting..."
	        exit 1
	    fi
	fi
	cd -
}


cast_error() {
	if (($? > 0)); then
    	printf '%s\n' "$1" >&2
    	exit 1
	fi
}


echo -n Password: 
read -s password

verify_cuda


## Conda environment
echo 'Checking if conda environment is installed'
conda --version
if (($? > 0)); then
    printf 'Installing conda'
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
	wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.27-Linux-x86_64.sh -O ~/miniconda.sh && \
	    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
	    rm ~/miniconda.sh
	export PATH=~/opt/conda/bin:$PATH

	alias conda="~/opt/conda/bin/conda"
else
	echo 'conda already installed'
fi

## Core rendering functionality
conda install -c menpo opencv -y &
P_opencv=$!
#cast_error 'Opencv installation failed'
conda install pytorch torchvision cuda80 -c soumith -y &
P_torch=$!
#cast_error 'Pytorch installation failed'


## Core multi channel GLFW
echo $password | sudo apt-get install libzmq3-dev libglew-dev libglm-dev
cast_error 'Opengl installation failed'
echo $password | sudo apt install mesa-common-dev libglu1-mesa-dev freeglut3-dev
cast_error 'Opengl addons installation failed'


wget https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip
unzip glfw-3.1.2.zip && rm glfw-3.1.2.zip
mv glfw-3.1.2 ./realenv/core/channels/external/glfw-3.1.2
mkdir ./realenv/core/channels/build
cd ./realenv/core/channels/build
cmake .. && make -j 10
cd -

## Core renderer
cd ./realenv/core/render/
wget https://www.dropbox.com/s/msd32wg144eew5r/coord.npy
pip install cython
bash build.sh
bash build_cuda.sh
python setup.py build_ext --inplace
cd -

## Data set
cd ./realenv/data
mkdir dataset
wget https://www.dropbox.com/s/gtg09zm5mwnvro8/11HB6XZSh1Q.zip
unzip -q 11HB6XZSh1Q.zip && rm 11HB6XZSh1Q.zip
mv 11HB6XZSh1Q dataset
cd -

## Physics Models
cd ./realenv/core/physics
wget https://www.dropbox.com/s/vb3pv4igllr39pi/models.zip
unzip -q models.zip && rm models.zip
cd -

wait $P_opencv $P_torch