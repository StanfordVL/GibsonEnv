## Physics Simulation Engine for Real Env
Zhiyang He

### Requirements
Python 3 environment (technically pybullet builds on python2, but I haven't been able to get it working)

Create necessary environment
```shell
conda create -n (env_name) python=3.5
pip install pybullet, zmq, transforms3d
```

Make proper obj file for OpenGL convention
```
python invert_obj --datapath ../data --model model_id
```


### Run the code
Simply running this will not do anything, because now physics renderer is tied together with view renderer. You should not be running standalone view renderer either
```shell
python render_physics.py --datapath ../data --model model_id
```


### Run everything at once
Open three different shells, and follow the sequence
```shell
## Shell One
cd /realenv/depth/depth_render
./depth_render --datapath ../../data -m 11HB6XZSh1Q

## Shell Two 
cd /realenv/dev
source activate (py2_universe_env)
python show_3d2_f.py --datapath ../data/ --idx 10

## Shell Three
cd /realenv/physics
source activate (py35_env)
python render_physics.py --datapath ../data --model 11HB6XZSh1Q
```
You should see OpenCV windows, as well as pybullet panel pop up.

To control the object, click on pybullet panel, then use the following keys. Pybullet sends its movement to
Opengl & OpenCV for rendering.

| w/q  | go forward |
| ------------- | ------------- |
| a  | go left  |
| s  | go backward  |
| d  | go right |
| z | go upward |
| c | go downward |
| u/j  | add/subtract roll |
| i/k  | add/subtract pitch |
| o/l | add/subtract yaw |
  
