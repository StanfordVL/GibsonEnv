#!/bin/bash

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


verify_conda() {
	## Conda environment
	echo 'Checking if conda environment is installed'
	conda --version
	if (($? > 0)); then
	    printf 'Installing conda'
	    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
		wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.21-Linux-x86_64.sh -O ~/miniconda.sh && \
		    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
		    rm ~/miniconda.sh
		export PATH=~/opt/conda/bin:$PATH

		alias conda="~/opt/conda/bin/conda"
	else
		echo 'conda already installed'
	fi
}

install() {
	set -e

	echo -n Password: 
	read -s password

	## Core rendering functionality
	#conda install -c menpo opencv -y
	pip install opencv-python			## python3
	conda install pytorch torchvision cuda80 -c soumith -y
	
	git clone https://github.com/openai/baselines.git
	cd baselines
	pip install -e .
	## need to remove one line from baseline
	cd -

	## Core multi channel GLFW
	echo $password | sudo -s apt-get update
	echo $password | sudo -s apt-get install libzmq3-dev libglew-dev libglm-dev libassimp-dev xorg-dev libglu1-mesa-dev libboost-dev -y
	echo $password | sudo -s apt install mesa-common-dev libglu1-mesa-dev freeglut3-dev libopenmpi-dev -y
	echo $password | sudo -s apt autoremove
	echo $password | sudo -s apt install cmake -y
	echo $password | sudo -s apt install golang libjpeg-turbo8-dev unzip -y
	echo $password | sudo -s apt install wmctrl xdotool -y

	## Core renderer
	echo $password | sudo -s apt install nvidia-cuda-toolkit -y	## Huge, 1121M

	build_local
	
	download_data
}

build_local() {
	## Core renderer
	if [ ! -d ./realenv/core/channels/external/glfw-3.1.2 ]; then
		wget --quiet https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip
		unzip glfw-3.1.2.zip && rm glfw-3.1.2.zip
		mv glfw-3.1.2 ./realenv/core/channels/external/glfw-3.1.2
	fi
	[ ! -d ./realenv/core/channels/build ] || rm -rf ./realenv/core/channels/build

	mkdir -p ./realenv/core/channels/build
	cd ./realenv/core/channels/build
	cmake .. && make clean && make -j 10
	cd -


	cd ./realenv/core/render/
	pip install cython
	bash build.sh
	bash build_cuda.sh
	python setup.py build_ext --inplace
	cd -
}

download_data () {
	## Data set
	cd ./realenv/data
	[ -d dataset ] || mkdir dataset
	[ ! -d ./realenv/core/physics/models ] || rm -rf ./realenv/core/physics/models
	

	## Psych building -1F, 919Mb
	if [ $dset_name="stanford_1" ] && [ ! -d dataset/BbxejD15Etk ]; then
		wget https://www.dropbox.com/s/zm112zosuxnx8by/BbxejD15Etk.zip
		unzip -q BbxejD15Etk.zip && rm BbxejD15Etk.zip
		mv BbxejD15Etk dataset
	fi

	## Psych building 1F, 794.2Mb
	if [ $dset_name="stanford_2" ] && [ ! -d dataset/sitktXish3E ]; then
		wget https://www.dropbox.com/s/bt4pctsnh3recs7/sitktXish3E.zip
		unzip -q sitktXish3E.zip && rm sitktXish3E.zip
		mv sitktXish3E dataset
	fi

	## Gates building 1F, 616.1Mb
	if [ $dset_name="stanford_3" ] && [ ! -d dataset/sRj553CTHiw ]; then
		wget https://www.dropbox.com/s/m9t9njmymwfvt6d/sRj553CTHiw.zip
		unzip -q sRj553CTHiw.zip && rm sRj553CTHiw.zip
		mv sRj553CTHiw dataset
	fi

	## Gates building 2F, 294.1Mb
	if [ $dset_name="stanford_4" ] && [ ! -d dataset/TVHnHa4MZwE ]; then
		wget https://www.dropbox.com/s/f9y73pafshhjlbu/TVHnHa4MZwE.zip
		unzip -q TVHnHa4MZwE.zip && rm TVHnHa4MZwE.zip
		mv TVHnHa4MZwE dataset
	fi


	if [ ! -d dataset/11HB6XZSh1Q ]; then
		wget https://www.dropbox.com/s/zplz70g7dxdcnii/11HB6XZSh1Q.zip
		unzip -q 11HB6XZSh1Q.zip && rm 11HB6XZSh1Q.zip
		mv 11HB6XZSh1Q dataset
	fi
	cd -

	## Physics Models
	if [ ! -d ./realenv/core/physics/models ]; then
		cd ./realenv/core/physics
		wget --quiet https://www.dropbox.com/s/3w9vxc8f071u1h0/models.zip
		unzip -q models.zip && rm models.zip
		cd -
	fi

	cd ./realenv/core/render/		
		if [ ! -f coord.npy ]; then
			wget --quiet https://www.dropbox.com/s/msd32wg144eew5r/coord.npy
		fi
		if [ ! -f model.pth ]; then
			wget --quiet https://www.dropbox.com/s/e7far9okgv7oq8p/model.pth
			wget --quiet https://www.dropbox.com/s/ywrkh66porlqcax/compG_epoch4_3000.pth
		fi
	cd -
		
	if [ -f realenv/data/*.pkl ]; then
		rm realenv/data/*.pkl
	fi
}


ec2_install_conda() {
    if [ ! -d ~/miniconda2 ]; then
     	wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.21-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b && rm ~/miniconda.sh
    fi
    export PATH=/home/ubuntu/miniconda2/bin:$PATH
    echo "PATH=/home/ubuntu/miniconda2/bin:$PATH" >> ~/.bashrc
    echo "Run this command in your aws terminal:"
    echo "	export PATH=/home/ubuntu/miniconda2/bin:$PATH"
}

hello() {
	echo "hello world"
}

subcommand=$1
dset_name=$2
case "$subcommand" in                                                                                
  "install")
	install
	;;
  "hello" )
	hello
	;;
  "ec2_install_conda")                                                           
    ec2_install_conda
    ;;
  "verify_cuda")
	verify_cuda
	;;
  "verify_conda")
	verify_conda
	;;
  "download_data")
	download_data
	;;
  "build_local")
	build_local
	;;
  *)                                                                 
    default "$@"                                       
    exit 1                                                                             
    ;;                                                                                 
esac 
