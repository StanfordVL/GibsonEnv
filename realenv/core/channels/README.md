## Universe Environment Multichannel Renderer
Support for real time multichannel input (depth, surface normal, etc), implemented using OpenGL. 

All external libraries have been included. You only need CMake (gcc version <=5) on your computer in order to build
![](https://github.com/fxia22/realenv/blob/depth_render/misc/depth_render.png)


### Steps

#### Install dependencies
```shell
sudo apt-get install libzmq3-dev libassimp-dev
sudo apt-get install libglew-dev
sudo apt-get install libglm-dev
sudo apt-get install xorg-dev 
sudo apt-get install mesa-common-dev
sudo apt-get install libglu1-mesa-dev freeglut3-dev

```




Download models: you *do not* have to download dataset separately for this module. Feel free to skip this step, or refer to https://github.com/fxia22/realenv/tree/physics/realenv/envs. If you want to download models for demonstration purpose, do the following
```shell
cd {realenv_root}/realenv/envs/channels/depth_render
wget https://www.dropbox.com/s/qc6sqqua2r2ht4w/16b32add7aa946f283740b9c1c1646c0.obj
wget https://www.dropbox.com/s/w7qdbzmo6pp0466/Q97jUzc1wSS_HIGH.obj
wget https://www.dropbox.com/s/gs7vhmanjgao9vy/suzanne.obj
```

Compile C code (this might take a minute or two)
```shell
cd {realenv_root}/realenv/envs/channels
mkdir build && cd build
cmake ..
make
```

Note: GCC Version, if you get the CMake complaint saying `CUDA on Ubuntu: unsupported GNU version! gcc versions later than 5 are not supported!`, you can try out the following solution

```shell
sudo apt install gcc-4.9 g++-4.9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 50 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
update-alternatives --list gcc
update-alternatives --set gcc /usr/bin/gcc-4.9
```


Finally, you're done. Although Multichannel Renderer is not supposed to be run on its own, you can do the following to get a sense of it. 
```
cd {realenv_root}/realenv/envs/channels/depth_render
./demo
```
