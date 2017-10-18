wget http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
bash NVIDIA-Linux-x86_64-367.57.run

sudo apt install xinit
#start x 
sudo apt-get install mesa-utils xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
sudo /usr/bin/X :0 &1
DISPLAY=:0 glxinfo
nvidia-smi
DISPLAY=:0 glxgears