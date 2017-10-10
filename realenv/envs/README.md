## Real Universe Environment

This folder has the source code for environment bundle, which includes viewer renderer, multichannel renderer and physics simulation engine

### Run everything at once
Open three different shells, and follow the sequence
```shell
## Shell One
cd {realenv_root}/realenv/envs/depth/depth_render
./depth_render --datapath ../data -m 11HB6XZSh1Q

## Shell Two 
cd {realenv_root}/realenv/envs/render
source activate (py2_universe_env)
python show_3d2.py --datapath ../data/ --idx 10

## Shell Three
cd {realenv_root}/realenv/envs/physics
source activate (py2_universe_env)
python render_physics.py --datapath ../data --model 11HB6XZSh1Q
```
You should see OpenCV windows, as well as pybullet panel pop up.

To control the object, click on pybullet panel, then use the following keys. Pybullet sends its movement to
Opengl & OpenCV for rendering.

| key  | action |
| ------------- | ------------- |
| w/q | go forward |
| a  | go left  |
| s  | go backward  |
| d  | go right |
| z | go upward |
| c | go downward |
| u/j  | add/subtract roll |
| i/k  | add/subtract pitch |
| o/l | add/subtract yaw |

Note that key w has conflict with pybullet hotkey.