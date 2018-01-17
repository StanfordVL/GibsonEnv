## Gibson release docker

Much of the Dockerfile is based on: https://github.com/thewtex/docker-opengl

For Gibson platform:

## Requirements:
	
	NVidia CUDA capable GPU, CUDA 7.0, CUDNN 5

### Install Docker
```shell
## https://github.com/moby/moby/issues/22371
sudo apt-get remove docker docker-engine docker.iols
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo docker run hello-world
```

### Install nvidia-driver 
```shell
(note: only 1.0.1 version works, if you have 2.0.0 installed it might 
cause conflict)
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
```

## Install

```shell
# Will create stanford/gibson:0.1
make build 

# Run graphical application
bash run.sh -r --env="APP=glxgears"
```

Instruction for setting up Gibson environment on GPU host
