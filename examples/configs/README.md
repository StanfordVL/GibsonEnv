Environment Configuration
=================
Each environment is configured with a `yaml` file. Examples of `yaml` files can be found in `examples/configs` folder. Parameters for the file is explained as below (take navigation environment for example):

```
envname: AntClimbEnv # Environment name, make sure it is the same as the class name of the environment
model_id: space7 # Scene id
target_orn: [0, 0, 3.14] # target orientation for navigating, the reference frame is world frame
target_pos: [-7, 2.6, -1.5] # target position for navigating, the reference frame is world frame
initial_orn: [0, 0, 3.14] # initial orientation for navigating
initial_pos: [-7, 2.6, 0.5] # initial position for navigating
fov: 1.57 # field of view for the camera
use_filler: true # use neural network filler or not, it is recommended to leave this argument true
display_ui: true # show pygame ui or not, if in a production environment (training), you need to turn this off
show_diagnostics: true # show dignostics overlaying on the RGB image
ui_num: 2 # how many ui components to show
ui_components: [RGB_FILLED, DEPTH] # which are the ui components, choose from [RGB_FILLED, DEPTH, NORMAL, SEMANTICS, RGB_PREFILLED]

output: [nonviz_sensor, rgb_filled, depth] # output of the environment to the robot
resolution: 512 # resolution of rgb/depth image

speed:
  timestep: 0.01 # timestep of simulation in seconds
  frameskip: 1 # how many frames to run simulation for one action

mode: gui # gui|headless, if in a production environment (training), you need to turn this to headless
verbose: false # show dignostics in terminal
```
