import os
import matplotlib.pyplot as plt
import csv
import numpy as np

## s"udo /usr/bin/X :0"
## DISPLAY=:0 python examples/train/train_husky_navigate_ppo1.py
## DISPLAY=:0 python examples/train/train_husky_navigate_ppo1.py --disable_filler

## scp -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@ec2-34-215-160-202.us-west-2.compute.amazonaws.com:/tmp/openai-2017-11-12-13-16-28-113773/0.monitor.csv
cmd = "scp -i /home/jerry/Dropbox/CVGL/universe.pem ubuntu@{}:/tmp/{}/{} {}"

aws_addr = []
log_dir  = []

'''
Universe husky navigate 1
RGB + SENSOR + COLLISION
'''
aws_addr.append("ec2-34-215-160-202.us-west-2.compute.amazonaws.com")
log_dir.append("openai-2017-11-12-13-43-46-948548")


'''
Universe husky navigate 2
RGB + NO SENSOR + COLLISION
        timesteps_per_actorbatch=1024,
        clip_param=0.2, entcoeff=0.001,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.995, lam=0.95,
'''
aws_addr.append("ec2-52-24-76-187.us-west-2.compute.amazonaws.com")
log_dir.append("openai-2017-11-12-14-31-31-175329")


'''
Universe husky navigate 3
RGB + NO SENSOR + COLLISION
        timesteps_per_actorbatch=1024,
        clip_param=0.2, entcoeff=0.0001,
        optim_epochs=10, optim_stepsize=3e-6, optim_batchsize=64,
        gamma=0.995, lam=0.95,
'''
aws_addr.append("ec2-52-42-249-133.us-west-2.compute.amazonaws.com")
log_dir.append("openai-2017-11-12-14-27-49-793285")


'''
Universe husky navigate 4
RGB + SENSOR + NO COLLISION
'''
aws_addr.append("ec2-52-38-25-41.us-west-2.compute.amazonaws.com")
log_dir.append("openai-2017-11-12-14-03-35-785424")



filename = "0.monitor.csv"
local_logs = []
for i in range(len(log_dir)):
    local_logs.append(str(i) + filename)

def download(index):
    full_cmd = cmd.format(aws_addr[index], log_dir[index], filename, local_logs[index])
    print(full_cmd)
    os.system(full_cmd)


def plot(index, axis, smooth_fc=6):
    with open(local_logs[index], 'r') as csvfile:
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
    all_times = num_rows[:, 2].tolist()
    t_range = np.arange(0, len(all_rewards), 1)
    if smooth_fc:
        axis[index].plot(t_range, all_rewards, '.', t_range, smooth_median(all_rewards, smooth_fc), '-')
    else:
        axis[index].plot(t_range, all_rewards, '.')


def smooth_median(rewards, factor=10):
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

def main():
    for i in range(len(log_dir)):
        download(i)

    f, axarr = plt.subplots(len(log_dir), sharex=True)
    for i in range(len(log_dir)):
        plot(i, axarr, 10)
    plt.show()

if __name__ == '__main__':
    main()
