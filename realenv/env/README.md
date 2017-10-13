## Real Universe Environment

This folder has the source code for environment bundle, which includes view renderer, multichannel renderer and physics simulation engine

### Requirement
You need CUDA 8.0 in order to run the environment.

### Run everything in a bundle
Current trainin environment in implemented in `simple_env.py`. It is a dead simple room exploration game with negative delta distance as reward. It's default to ru with model `11HB6XZSh1Q`. We provide a sample agent `random_husky.py` to interact with this environment. To start, run the following command
```shell
source activate (py2_universe_env)
python random_husky.py
```
You can switch between agent/human mode by setting `human=True/False` when initializing environment object. You can also set `debug=True/False` to run with or without graphical interface.

### Components
The above environment has three components: view renderer, multichannel renderer and physics simulation engine. A built-in add on is a scoreboard implemented with matplotlib.

#### Test each component individually
Multichannel renderer currently only supports depth rendering. In the future we will provide surface normal, semantics and arbitrary meta information.
```shell
## Shell One
cd depth/depth_render
./depth_render --datapath ../../data -m 11HB6XZSh1Q
```

View renderer. 
```shell
## Shell Two 
cd render
source activate (py2_universe_env)
python show_3d2.py --datapath ../data/ --idx 10
```

Physics engine
```shell
## Shell Three
cd physics
source activate (py2_universe_env)
python render_physics.py --datapath ../data --model 11HB6XZSh1Q
```

If you want to test out how they work together in a combined way, you don't have to open up three separate terminals. Run
```shell
source acitvate (py2_universe_env)
python simple_env.py
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