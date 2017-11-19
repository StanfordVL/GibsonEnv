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
aws_names.append("Depth & Sensor, zero init random, lr=3E-6")

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
#aws_names.append("RGBD & Sensor, small init random, small lr")


'''
Universe ant climb 6
ec2-34-216-35-8.us-west-2.compute.amazonaws.com
'''
aws_addr.append("ec2-34-216-35-8.us-west-2.compute.amazonaws.com")
aws_names.append("Depth & Sensor, tiny init random, lr=3E-6")

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
Universe husky navigate 4
RGB + SENSOR + NO COLLISION
'''
aws_addr.append("ec2-52-38-25-41.us-west-2.compute.amazonaws.com")
log_dir.append("openai-2017-11-12-14-03-35-785424")



scp_cmd = "scp -i /home/zhiyang/Dropbox/CVGL/universe.pem ubuntu@{}:/tmp/{}/{} {}"

def download(index):
    full_cmd = cmd.format(aws_addr[index], log_dir[index], filename, local_logs[index])
    print(full_cmd)
    os.system(full_cmd)


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
    all_times = num_rows[:, 2].tolist()
    t_range = np.arange(0, len(all_rewards), 1)
    step_length = sum(all_steps)
    total_time = num_rows[-1, 2]
    time_length = sum(all_times)

    smoothed_reward = smooth_median(all_rewards)
    smoothed_range = np.arange(0, len(smoothed_reward), 1)

    #axis[index].plot(t_range, all_rewards, '.', smoothed_range, smoothed_reward, '-')
    axis[index].plot(smoothed_range, smoothed_reward, '-')
    axis[index].text(.5, .9, aws_names[index] + " ts: {}".format(step_length), horizontalalignment='center',
    transform=axis[index].transAxes)
    axis[index].text(.5, .1, "Time ts: {}".format(time_length), horizontalalignment='center',
    transform=axis[index].transAxes)
    

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
    f, axarr = plt.subplots(len(file_names), sharex=True)
    for i in range(len(file_names)):
        plot(i, axarr)
    plt.show()

    f, axarr = plt.subplots(len(log_dir), sharex=True)
    for i in range(len(log_dir)):
        plot(i, axarr, 10)
    plt.show()

if __name__ == '__main__':
    LOCAL = True
    SMOOTH = True
    SMOOTH_FC = 10000

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
    main()
