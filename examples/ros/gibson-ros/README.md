Gibson ros binding
============
 
This is a ros package that contains some examples of using Gibson Env with ros navigation stack. 
 
## Preparation
 
1. Install ROS: in this package, we use navigation stack from ros kinetic. Please follow the [instructions](http://wiki.ros.org/kinetic/Installation/Ubuntu).  
2. Install ROS dependencies:
```bash
sudo apt-get install ros-kinetic-turtlebot-navigation
sudo apt-get install ros-kinetic-nav2d
sudo apt-get install ros-kinetic-map-server
```
3. Install gibson __from source__ following [installation guide](../../README.md) in __python2.7__. However, as ros only supports `python2.7` at the moment, you need to create python2.7 virtual environment instead of python3.5.
4. If you use annaconda for setting up python environment, some tweaks of `PATH` and `PYTHONPATH` variable are required to avoid conflict. In particular:
	1. For `PATH`: conda related needs to be removed from `PATH`
	```bash
	echo $PATH | grep -oP "[^:;]+" | grep conda	## Remove these paths from $PATH
	```
	2. For `PYTHONPATH`: `/usr/lib/python2.7/dist-packages/`, `/opt/ros/kinetic/lib/python2.7/dist-packages`(ros python libraries), `<anaconda installation root>/anaconda2/envs/py27/lib/python2.7/site-packages`(gibson dependencies) and `<gibson root>/gibson` need to be in `PYTHONPATH`.
5. Finally, copy (or soft link) gibson-ros folder to your `catkin_ws/src` and run catkin_make to index gibson-ros package.
```bash
ln -s examples/ros/gibson-ros/ ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make && cd -
```

## Sanity check 

```bash
which python #should give /usr/bin/python 
python -c 'import gibson, rospy, rospkg' #you should be able to do those without errors.
```

## Running
1. Prepare ROS environment
```bash
source /opt/ros/kinetic/setup.bash
source <catkin-workspace-root>/catkin_ws/devel/setup.bash
```
2. Repeat step 3 from Preparation, sanitize `PATH` and `PYTHONPATH`
3. Enjoy
```bash
roslaunch gibson-ros turtlebot_gmapping.launch #Run gmapping
roslaunch gibson-ros turtlebot_hector_mapping.launch #Run hector mapping
roslaunch gibson-ros turtlebot_navigation.launch #Run the navigation stack, we have provided the map
```

The following screenshot is captured when running the gmapping example.

![](misc/slam.png)
