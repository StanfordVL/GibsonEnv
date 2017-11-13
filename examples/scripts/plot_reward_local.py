import os
import matplotlib.pyplot as plt
import csv
import numpy as np


aws_addr = []
log_dir  = []


root_dir = "/tmp"
all_log_dirs = sorted(os.listdir(root_dir))
all_log_dirs = [os.path.join(root_dir, d) for d in all_log_dirs if "openai" in d]


filename = "0.monitor.csv"
local_logs = [os.path.join(all_log_dirs[-1], filename)]
log_dir = [all_log_dirs[-1]]

for i in range(len(log_dir)):
    local_logs.append(str(i) + filename)

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

def smooth_max(rewards, factor=10):
    rew_median = []
    i = 0
    while i < len(rewards):
        cur_list = rewards[i: min(i + factor, len(rewards))]
        while len(cur_list) < factor:
            cur_list.append(rewards[-1]) 
        i = i + 1
        rew_median.append(np.max(cur_list))
    return rew_median

def plot(index, smooth_fc=smooth_median):
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

    row_index = num_rows[:, 0] > -200   ## Avoid plotting too low points

    all_rewards = num_rows[row_index, 0].tolist()
    all_times = num_rows[row_index, 1].tolist()
    t_range = np.arange(0, len(all_rewards), 1)
    print("Total number of steps:", sum(all_times))
    plt.plot(t_range, all_rewards, '.', t_range, smooth_fc(all_rewards, args.smooth), '-')
    

def smooth(rewards, factor=10):
    smoothed_rew = []
    for i in range(0, len(rewards) - factor + 1):
        smoothed_rew.append(sum(rewards[i:i + factor]) / factor)
    return smoothed_rew

def main():
    plot(0, smooth_median)
    #plot(0, smooth_max)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smooth', type=int, default=50)
    args = parser.parse_args()
    main()
