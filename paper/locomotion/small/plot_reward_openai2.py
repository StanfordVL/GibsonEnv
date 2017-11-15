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
#aws_names.append("RGBD & Sensor, small init random, small lr")


'''
Universe ant climb 2
ssh -i universe.pem ubuntu@ec2-52-24-76-187.us-west-2.compute.amazonaws.com
'''
aws_addr.append("ec2-52-24-76-187.us-west-2.compute.amazonaws.com")
aws_names.append("RGBD & Sensor, small init random, large lr")


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
aws_addr.append("ec2-52-38-25-41.us-west-2.compute.amazonaws.com")
aws_names.append("Depth & Sensor, small init random, large lr")


'''
Universe ant climb 5
ec2-34-214-189-116.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-214-189-116.us-west-2.compute.amazonaws.com")
#aws_names.append("RGBD & Sensor, small init random, small lr")


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
#aws_names.append("Depth & Sensor, large random, small lr")


'''
Universe ant climb 10
ec2-34-212-248-24.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-212-248-24.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, large random, small lr")

'''
Universe ant climb 11
ec2-34-210-182-251.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-210-182-251.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, tiny init random, small lr")


'''
Universe ant climb 12
ec2-34-215-146-153.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-34-215-146-153.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, tiny init random, large lr")

'''
13
ec2-52-32-82-119.us-west-2.compute.amazonaws.com
'''
#aws_addr.append("ec2-52-32-82-119.us-west-2.compute.amazonaws.com")
#aws_names.append("Depth & Sensor, small random, large lr")



aws_dirs = []
file_names = []

remote_filename = "0.monitor.csv"
name_template = "monitor_{}_{}.csv"

scp_cmd = "scp -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@{}:/tmp/{}/{} {}"

def download(index):
    full_cmd = scp_cmd.format(aws_addr[index], aws_dirs[index], remote_filename, file_names[index])
    #print(full_cmd)
    os.system(full_cmd)


def get_latest_openai_dir(host):
    ## ssh -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com ls -l /tmp/
    # ls_cmd  = "ssh -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@{} ls -l /tmp/"
    COMMAND="ls -l /tmp/"
    ssh = subprocess.Popen(["ssh", "-i", "/home/jerry/Dropbox/CVGL/universe.pem", "ubuntu@%s" % host, COMMAND],
       shell=False,
       stdout=subprocess.PIPE,
       stderr=subprocess.PIPE)
    results = ssh.stdout.readlines()
    results = [str(r,'utf-8') for r in results]
    results = [r.split()[-1].strip() for r in results if "openai" in r]
    results = sorted(results)
    return results[-1]


def plot(index, axis):
    with open(file_names[index], 'r') as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        str_rows = []
        num_rows = []
        for row in csv_rows:
            str_rows.append(row[0])
        for row in str_rows:
            try:
                num_rows.append([float(num) for num in row.split(',')])
            except:
                continue
        num_rows = np.array(num_rows)

    all_rewards = num_rows[:, 0].tolist()
    all_steps = num_rows[:, 1].tolist()
    t_range = np.arange(0, len(all_rewards), 1)
    step_length = sum(all_steps)
    total_time = num_rows[-1, 2]
    time_length = sum(all_times)
    if SMOOTH:
        axis[index].plot(t_range, all_rewards, '.', t_range, smooth_mean(all_rewards, SMOOTH_FC), '-')
        axis[index].text(.5, .9, aws_names[index] + " ts: {}".format(step_length), horizontalalignment='center',
        transform=axis[index].transAxes)
        axis[index].text(.5, .1, "Time ts: {}".format(time_length), horizontalalignment='center',
        transform=axis[index].transAxes)
    else:
        axis[index].plot(t_range, all_rewards, '.')
        axis[index].text(.5, .9, aws_names[index] + " ts: {}".format(step_length), horizontalalignment='center',
        transform=axis[index].transAxes)
        axis[index].text(.5, .1, "Time ts: {}".format(time_length), horizontalalignment='center',
        transform=axis[index].transAxes)


def get_smooth(index):
    with open(file_names[index], 'r') as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        str_rows = []
        num_rows = []
        for row in csv_rows:
            str_rows.append(row[0])
        for row in str_rows:
            try:
                num_rows.append([float(num) for num in row.split(',')])
            except:
                continue
        num_rows = np.array(num_rows)

    all_rewards = num_rows[:, 0].tolist()
    total_time = num_rows[-1, 2]
    all_steps = num_rows[:, 1].tolist()
    #print("Total time", total_time, "averge step", np.mean(all_steps))
    return smooth_median(all_rewards)



def smooth_median(rewards):
    rew_median = []
    i = 0
    while i < len(rewards) - SMOOTH_FC:
        cur_list = rewards[i: min(i + SMOOTH_FC, len(rewards))]
        while len(cur_list) < SMOOTH_FC:
            cur_list.append(rewards[-1]) 
        i = i + 1
        rew_median.append(np.median(cur_list))
    return rew_median

def smooth_mean(rewards):
    rew_mean = []
    i = 0
    while i < len(rewards) - SMOOTH_FC:
        cur_list = rewards[i: min(i + SMOOTH_FC, len(rewards))]
        while len(cur_list) < SMOOTH_FC:
            cur_list.append(rewards[-1]) 
        i = i + 1
        rew_mean.append(np.mean(cur_list))
    return rew_mean

def smooth(rewards):
    smoothed_rew = []
    for i in range(0, len(rewards) - SMOOTH_FC + 1):
        smoothed_rew.append(sum(rewards[i:i + SMOOTH_FC]) / SMOOTH_FC)
    return smoothed_rew

def main():
    f, axarr = plt.subplots(len(file_names), sharex=True)
    for i in range(len(file_names)):
        plot(i, axarr, 10)
    plt.show()

def main2():
    LENGTH_CAP = 1000000000

    smoothed = []
    lengths = [] 
    smoothed_names = []
    for i in range(len(file_names)):
        smooth_i = get_smooth(i)
        #smooth_i = smooth_mean(smooth_i)
        if len(smooth_i) < LENGTH_CAP:
            smoothed.append(smooth_i)
            lengths.append(len(smooth_i))
            smoothed_names.append(aws_names[i])
    max_time_length = max([len(series) for series in smoothed])    
    
    legends = []
    for i in range(len(smoothed)):
        patch, = plt.plot(np.arange(0, lengths[i], 1), smoothed[i], label=smoothed_names[i])
        legends.append(patch)
    plt.legend(handles=legends)
    plt.show()



if __name__ == '__main__':
    LOCAL = False
    SMOOTH = True
    SMOOTH_FC = 100

    if LOCAL:
        for i in range(len(aws_addr)):
            host = aws_addr[i]
            remote_dir = get_latest_openai_dir(host)
            file_names.append(name_template.format(host, remote_dir))
    else:
        for i in range(len(aws_addr)):
            host = aws_addr[i]
            remote_dir = get_latest_openai_dir(host)
            print(host, remote_dir)
            aws_dirs.append(remote_dir)
            file_names.append(name_template.format(host, remote_dir))
            download(i)
    file_names = file_names[:]
    main2()
