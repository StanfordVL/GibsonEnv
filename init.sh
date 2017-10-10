./realenv/envs/build.sh

x_display=1
vncserver :$x_display -depth 24 -geometry 512x256 2>&1 &
echo "Using display: x_display=$x_display"

export DISPLAY=":$x_display"

python realenv/envs/show_3d.py --dataroot ./data/viewsyn_tiny/ --idx 3 --model compG_epoch3_10000.pth
