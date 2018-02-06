# Gibson Environment for Training Real World AI
You shouldn't play video games all day, so shouldn't your AI. In this project we build a virtual environment that offers real world experience. You can think of it like [The Matrix](https://www.youtube.com/watch?v=3Ep_rnYweaI).

## Note
This is a 0.1.0 beta release, bug reports are welcome. 

### Installation

The minimal system requirements are the following:

- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

#### Dependencies

We use docker to distribute our software, you need to install [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. 

Run `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` to verify your installation. 

#### Download data

Download data from [here](https://drive.google.com/open?id=1jV-UN4ePwsE9XYv8m4YbNiGxRI_WWpW0) and put `assets.tar.gz` it in `gibson/assets` folder.


#### Quick installation (docker)

1. Build your own docker image (recommended)
```bash
git clone -b dev https://github.com/fxia22/gibson.git 
cd gibson
#download data file from https://drive.google.com/open?id=1jV-UN4ePwsE9XYv8m4YbNiGxRI_WWpW0 and put it into gibson/assets folder
./build.sh decompress_data ### Download data outside docker, in case docker images need to be rebuilt
docker build . -t gibson
```
If the installation is successful, you should be able to run `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gibson` to create a container.


2. Or pull from our docker image
```bash
docker pull xf1280/gibson:latest
```

#### Build from source

First, make sure you have Nvidia driver and CUDA installed. If you install from source, CUDA 9 is not necessary, as that is for nvidia-docker 2.0. Then, let's install some dependencies:

```bash
apt-get update 
apt-get install libglew-dev libglm-dev libassimp-dev xorg-dev libglu1-mesa-dev libboost-dev \
		mesa-common-dev freeglut3-dev libopenmpi-dev cmake golang libjpeg-turbo8-dev wmctrl \ 
		xdotool libzmq3-dev zlib1g-dev\
```	

Install required deep learning libraries: Using python3.5 is recommended. You can create a python3.5 environment first. 

```bash
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
pip install tensorflow==1.3
```
Clone the repository, download data and build
```bash
git clone -b dev https://github.com/fxia22/gibson.git 
cd gibson
#download data from google drive
./build.sh decompress_data ### decompress data 
./build.sh build_local ### build C++ and CUDA files
pip install -e . ### Install python libraries
```

Install OpenAI baselines if you need to run training demo.

```bash
git clone https://github.com/fxia22/baselines.git
pip install -e baselines
```


## Demo

After getting into the docker container, you can run a few demos. You might need to run `xhost +local:root` to enable display. If you installed from source, you can run those directly. 

```bash
python examples/demo/play_husky_sensor.py ### Use ASWD to control a car to navigate around gates

python examples/demo/play_husky_camera.py ### Use ASWD to control a car to navigate around gates, with camera output

python examples/train/train_husky_navigate_ppo2.py --resolution NORMAL ### Use PPO2 to train a car to navigate down the hall way in gates based on visual input

###More to come!

```


