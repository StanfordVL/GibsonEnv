import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import subprocess
import sys


#sudo service lightdm stop
#sudo pkill Xorg
#sudo /usr/bin/X :0 &
## DISPLAY=:0 python examples/train/train_ant_climb_ppo1.py --mode RGBD --resolution SMALL
## DISPLAY=:0 python examples/train/train_ant_climb_ppo1.py --mode DEPTH --resolution SMALL
## DISPLAY=:0 python examples/train/train_ant_climb_ppo1.py --mode SENSOR

## scp -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com:/tmp/openai-2017-11-12-13-16-28-113773/0.monitor.csv
## ssh -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com ls -l /tmp/


aws_addr = []
aws_names = []

'''
Universe ant climb 1
ssh -i universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-215-160-202.us-west-2.compute.amazonaws.com")
#aws_names.append("RGB & Sensor, small init random, small lr")


'''
Universe ant climb 2
ssh -i universe.pem ubuntu@ec2-52-24-76-187.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-24-76-187.us-west-2.compute.amazonaws.com")
#aws_names.append("RGBD & Sensor, small init random, large lr")


'''
Universe ant climb 3
ec2-52-42-249-133.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-42-249-133.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, small init random, small lr")

'''
Universe ant climb 4
ec2-52-38-25-41.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-38-25-41.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, small init random, large lr")


'''
Universe ant climb 5
ec2-34-214-189-116.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-214-189-116.us-west-2.compute.amazonaws.com")
#aws_names.append("RGBD & Sensor, large init random, small lr")


'''
Universe ant climb 6
ec2-34-216-35-8.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-216-35-8.us-west-2.compute.amazonaws.com")
#aws_names.append("RGBD & Sensor, large init random, large lr")

'''
Universe ant climb 7
ec2-34-213-149-40.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-213-149-40.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, large init random, small lr")


'''
Universe ant climb 8
ec2-52-89-201-217.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-89-201-217.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, large init random, large lr")


'''
Universe ant climb 9
ec2-52-38-59-13.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-38-59-13.us-west-2.compute.amazonaws.com")
#aws_names.append("Sensor only, small lr")


'''
Universe ant climb 10
ec2-34-212-248-24.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-212-248-24.us-west-2.compute.amazonaws.com")
#aws_names.append("Sensor only, large lr")

'''
Universe ant climb 11
ec2-34-210-182-251.us-west-2.compute.amazonaws.com
'''
#aws_names.append("Depth & Sensor, tiny init random")


'''
Universe ant climb 12
ec2-34-215-146-153.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-215-146-153.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & only, tinier init random")


'''
13
ec2-52-32-82-119.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-32-82-119.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & only, small lr,  small random")



aws_dirs = []
local_dirs = []
aws_locals = []

<<<<<<< HEAD
mdl_cmd = "scp -i /home/zhiyang/Dropbox/CVGL/universe.pem -r ubuntu@{}:/home/ubuntu/realenv/examples/train/models/latest {}"
=======
mdl_cmd = "scp -i /home/jerry/Dropbox/CVGL/universe.pem -r ubuntu@{}:/home/ubuntu/realenv/examples/train/models/latest {}"
>>>>>>> 90d9f5fed89123ce8bfac67fb39d51426b6dd7e1

def get_latest_openai_dir(host):
    ## ssh -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com ls -l /tmp/
    # ls_cmd  = "ssh -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@{} ls -l /tmp/"
    COMMAND="ls -l /tmp/"
    ssh = subprocess.Popen(["ssh", "-i", "/home/zhiyang/Dropbox/CVGL/universe.pem", "ubuntu@%s" % host, COMMAND],
       shell=False,
       stdout=subprocess.PIPE,
       stderr=subprocess.PIPE)
    results = ssh.stdout.readlines()
    results = [str(r,'utf-8') for r in results]
    results = [r.split()[-1].strip() for r in results if "openai" in r]
    results = sorted(results)
    return results[-1]


def download_logs(index):
    scp_cmd = "scp -i /home/zhiyang/Dropbox/CVGL/universe.pem -r ubuntu@{}:/tmp/{}/ {}/"
    local_csv = os.path.join(aws_locals[index], "0.monitor.csv")
    full_cmd = scp_cmd.format(aws_addr[index], aws_dirs[index], local_dirs[index])
    #print(full_cmd)
    os.system(full_cmd)

def main():
    for i in range(len(aws_addr)):
        host = aws_addr[i]
        #local_dir = local_dirs[i]
        local_dir = "."

        aws_cmd = mdl_cmd.format(host, local_dir)
        os.system(aws_cmd)
        download_logs(i)

if __name__ == '__main__':
    for i in range(len(aws_addr)):
        host = aws_addr[i]
        remote_dir = get_latest_openai_dir(host)
        aws_dirs.append(remote_dir)

<<<<<<< HEAD
        aws_local = os.path.join("/home/zhiyang/Desktop/realenv/aws", host[:host.find(".")])
=======
        aws_local = os.path.join("/home/jerry/Desktop/realenv/models", host[:host.find(".")])
        '''
>>>>>>> 90d9f5fed89123ce8bfac67fb39d51426b6dd7e1
        if not os.path.isdir(aws_local):
            os.mkdir(aws_local)
        aws_locals.append(aws_local)
        local_dir = os.path.join(aws_local, remote_dir)
        if not os.path.isdir(local_dir):
<<<<<<< HEAD
            os.mkdir(local_dir)
        local_dirs.append(local_dir)
=======
            local_dirs.append(local_dir)
        '''
>>>>>>> 90d9f5fed89123ce8bfac67fb39d51426b6dd7e1
    main()
