./realenv/envs/build.sh

x_display=1
Xvnc4 :$x_display -PasswordFile pw -Zliblevel 0 -depth 24 -geometry 512x256 2>&1 &
echo "Using display: x_display=$x_display"

export DISPLAY=":$x_display"
xsetroot -solid grey -cursor_name left_ptr

python realenv/envs/show_3d.py --dataroot ~/Development/data/viewsyn_tiny/ --idx 3 --model compG_epoch3_10000.pth
