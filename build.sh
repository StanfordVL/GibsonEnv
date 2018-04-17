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
    pip install opencv-python            ## python3
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
    echo $password | sudo -s apt install libfreeimageplus3 libfreeimageplus-dev libfreeimage3 libfreeimage-dev -y

    ## Core renderer
    echo $password | sudo -s apt install nvidia-cuda-toolkit -y    ## Huge, 1121M

    build_local

    download_data
}

build_local() {
    ## Core renderer
    if [ ! -d ./gibson/core/channels/external/glfw-3.1.2 ]; then
        wget --quiet https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip
        unzip glfw-3.1.2.zip && rm glfw-3.1.2.zip
        mv glfw-3.1.2 ./gibson/core/channels/external/glfw-3.1.2
    fi
    [ ! -d ./gibson/core/channels/build ] || rm -rf ./gibson/core/channels/build

    mkdir -p ./gibson/core/channels/build
    cd ./gibson/core/channels/build
    cmake .. && make clean && make -j 10
    cd -


    cd ./gibson/core/render/
    bash build_cuda.sh
    cd -
}

decompress_data () {
    cd gibson
    tar -zxf assets.tar.gz
    rm assets.tar.gz
    if [ -f "gibson/assets/*.pkl" ]; then
        rm gibson/assets/*.pkl
    fi
    cd -
}


ec2_install_conda() {
    if [ ! -d ~/miniconda2 ]; then
         wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.21-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b && rm ~/miniconda.sh
    fi
    export PATH=/home/ubuntu/miniconda2/bin:$PATH
    echo "PATH=/home/ubuntu/miniconda2/bin:$PATH" >> ~/.bashrc
    echo "Run this command in your aws terminal:"
    echo "    export PATH=/home/ubuntu/miniconda2/bin:$PATH"
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
  "decompress_data")
    decompress_data
    ;;
  "build_local")
    build_local
    ;;
  *)                                                                 
    default "$@"                                       
    exit 1                                                                             
    ;;                                                                                 
esac 
