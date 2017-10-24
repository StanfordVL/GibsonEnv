## https://stackoverflow.com/questions/19856192/run-opengl-on-aws-gpu-instances-with-centos

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential

sudo apt-get install nvidia-375
#wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
#sudo bash NVIDIA-Linux-x86_64-367.57.run


#wget http://us.download.nvidia.com/tesla/384.66/nvidia-diag-driver-local-repo-ubuntu1604-384.66_1.0-1_amd64.deb

## Reboot now

#sudo apt install xinit
#start x
sudo apt-get install mesa-utils xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev -y
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

###
#  Expect
# Using X configuration file: "/etc/X11/xorg.conf".
# Backed up file '/etc/X11/xorg.conf' as '/etc/X11/xorg.conf.backup'
# New X configuration file written to '/etc/X11/xorg.conf'
###

sudo /usr/bin/X :0 &
DISPLAY=:0 glxinfo
nvidia-smi
DISPLAY=:0 glxgears




Correct log:

pybullet build time: Oct 18 2017 21:28:40
Processing the data:
Total 1 scenes 0 train 1 test
./depth_render --modelpath /home/ubuntu/realenv/realenv/data/dataset/11HB6XZSh1Q
opened this display, default 0
Error callback
X11: RandR gamma ramp support seems broken
10008
Error callbackg Environment ] |                                         | (ETA:  --:--:--) 
X11: RandR monitor support seems broken
10008
Compiling shader : ./StandardShadingRTT.vertexshader
Compiling shader : ./StandardShadingRTT.fragmentshader
Linking program
Loading OBJ file /home/ubuntu/realenv/realenv/data/dataset/11HB6XZSh1Q/modeldata/out_res.obj...
size of temp vertices 243747, vertex indices 685860 out vertices 685860  | (ETA:  0:00:35) 
Size of output vertex vector 