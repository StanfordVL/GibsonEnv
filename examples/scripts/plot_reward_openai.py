import os
import matplotlib.pyplot as plt
import csv
import numpy as np

## sudo /usr/bin/X :0
## DISPLAY=:0 python examples/train/train_husky_navigate_ppo1.py
## DISPLAY=:0 python examples/train/train_husky_navigate_ppo1.py --disable_filler

## scp -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com:/tmp/openai-2017-11-12-13-16-28-113773/0.monitor.csv
cmd = "scp -i /home/jerry/Dropbox/CVGL/universe.pem {}:/tmp/{}/{} ."


## Universe husky navigate 1
##  RGB + SENSOR + COLLISION
aws_addr = "ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com"
#log_dir  = "openai-2017-11-12-13-16-28-113773"
#log_dir  = /tmp/openai-2017-11-12-13-43-46-948548



## Universe husky navigate 4
##  RGB + SENSOR + NO COLLISION


filename = "0.monitor.csv"

def download():
    print(cmd.format(aws_addr, log_dir, filename))
    os.system(cmd.format(aws_addr, log_dir, filename))


def plot(smooth_fc=3):
    with open(filename, 'r') as csvfile:
        csv_rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        str_rows = []
        num_rows = []
        for row in csv_rows:
            str_rows.append(row[0])
        for row in str_rows[1:]:
            num_rows.append([float(num) for num in row.split(',')])
        num_rows = np.array(num_rows)

    all_rewards = num_rows[:, 0].tolist()
    all_times = num_rows[:, 2].tolist()
    print(all_rewards)    
    t_range = np.arange(0, len(all_rewards), 1)
    if smooth_fc:
        plt.plot(t_range, all_rewards, '.', t_range, smooth_median(all_rewards, 3), '-')
    else:
        plt.plot(t_range, all_rewards, '.')
    plt.show()

def smooth_median(rewards, factor=5):
    rew_median = []
    i = 0
    while i < len(rewards):
        cur_list = rewards[i: min(i + factor, len(rewards))]
        while len(cur_list) < factor:
            cur_list.append(rewards[-1]) 
        i = i + 1
        rew_median.append(np.median(cur_list))
    return rew_median

def smooth(rewards, factor=30):
    smoothed_rew = []
    for i in range(0, len(rewards) - factor + 1):
        smoothed_rew.append(sum(rewards[i:i + factor]) / factor)
    return smoothed_rew



if __name__ == '__main__':
    #download()
    plot(smooth_median)
