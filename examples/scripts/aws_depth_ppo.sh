## Starting
#DISPLAY=:0 python examples/train/train_husky_navigate_ppo1.py --mode DEPTH --resolution SMALL --save_name aws_train_husky_navigate_depth_ppo1 --disable_filler > aws_train_husky_navigate_depth_ppo1.log

## Downloading
#scp -i ~/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com:/home/ubuntu/realenv/examples/train/models/aws_train_husky_navigate_depth_ppo1_1200* ~/Desktop/realenv/*

## Enjoying
#python examples/train/enjoy_husky_navigate_ppo.py --reload_name aws_train_husky_navigate_depth_ppo1_1200.model --mode DEPTH --resolution SMALL --disable_filler 

scp -i ~/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com:/home/ubuntu/realenv/aws_train_husky_navigate_depth_ppo1.log aws_train_husky_navigate_depth_ppo1.log

python examples/scripts/plot_reward.py --file aws_train_husky_navigate_depth_ppo1.log --smooth 6