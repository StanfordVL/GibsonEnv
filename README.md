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

#### Quick installation (docker)

1. Build your own docker image (recommended)
```bash
git clone -b dev https://github.com/fxia22/gibson.git 
./build.sh download_data ### Download data outside docker, in case docker images need to be rebuilt
docker build . -t gibson
```
If the installation is successful, you should be able to run `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gibson` to create a container.


2. Download our docker image
TBA

#### Build from source
TBA


## Demo
TBA


