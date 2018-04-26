Gibson ros binding
============
 
This is a ros package that contains some examples of using Gibson Env with ros navigation stack. 
 
## Setup
 
1. In this package, we use navigation stack from ros kinect, follow the [instruction](http://wiki.ros.org/kinetic/Installation/Ubuntu) to install ros first.  
2. Install gibson __from source__ following [installation guide](../../README.md). However, as ros only supports `python2.7` at the moment, you need to create python2.7 virtual environmen instead of python3.5.
3. Some minor tweak of `PATH` and `PYTHONPATH` variable maybe required, in particular, `<anaconda installation root>/anaconda/bin` needs to be removed from `PATH`. `/usr/lib/python2.7/dist-packages/`, `/opt/ros/kinetic/lib/python2.7/dist-packages`(ros python libraries),
`<anaconda installation root>/anaconda2/envs/py27/lib/python2.7/site-packages`(gibson dependencies) and `<gibson root>/gibson` need to be in `PYTHONPATH`.
4. Finally, copy (or soft link) gibson-ros folder to your `catkin_ws/src` and run catkin_make to index gibson-ros package.

A sanity check of installation is to run `which python` and it should give `/usr/bin/python`. Then try to import `rospkg`, `rospy` and `gibson`, and you should be able to do those without errors.

## Running
```bash
roslaunch gibson-ros turtlebot_gmapping.launch #Run gmapping
roslaunch gibson-ros turtlebot_hector_mapping.launch #Run hector mapping
roslaunch gibson-ros turtlebot_navigation.launch #Run the navigation stack, we have provided the map
```


