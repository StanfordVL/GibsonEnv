## Setting up the OpenGL depth renderer
All external libraries have been included. You only need CMake on your computer in order to build
![](https://github.com/fxia22/realenv/blob/depth_render/misc/depth_render.png)

### Steps

Install dependencies
```shell
sudo apt install mesa-common-dev
sudo apt install libglu1-mesa-dev freeglut3-dev
```

Download models
```shell
cd {PROJECT_ROOT_PATH}/depth/depth_render
wget https://www.dropbox.com/s/qc6sqqua2r2ht4w/16b32add7aa946f283740b9c1c1646c0.obj
wget https://www.dropbox.com/s/w7qdbzmo6pp0466/Q97jUzc1wSS_HIGH.obj
wget https://www.dropbox.com/s/gs7vhmanjgao9vy/suzanne.obj
```

Compile C code (this might take a minute or two)
```shell
cd {PROJECT_ROOT_PATH}/depth
mkdir build && cd build
cmake ..
make
```

Run the code
```
./launch-depth_render.sh
```
