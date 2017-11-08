#python examples/train/train_husky_navigate_ppo1.py --mode RGBD --resolution SMALL > aws_train_husky_navigate_rgbd_ppo1.log

scp -i ~/Dropbox/CVGL/universe.pem ubuntu@ec2-52-41-31-139.us-west-2.compute.amazonaws.com:/home/ubuntu/realenv/aws_train_husky_navigate_rgbd_ppo1.log aws_train_husky_navigate_rgbd_ppo1.log

python examples/scripts/plot_reward.py --file aws_train_husky_navigate_rgbd_ppo1.log --smooth 1