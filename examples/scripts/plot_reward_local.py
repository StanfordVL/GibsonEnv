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
local_logs = [os.path.join(all_log_dirs[-1], filename), os.path.join(all_log_dirs[-2], filename)]
log_dir = [all_log_dirs[-1]]
print(log_dir)

for i in range(len(log_dir)):
    local_logs.append(str(i) + filename)


def smooth_median(rewards):
    rew_median = []
    i = 0
    while i < len(rewards) - MEDIAN_FC:
        cur_list = rewards[i: min(i + MEDIAN_FC, len(rewards))]
        while len(cur_list) < MEDIAN_FC:
            cur_list.append(rewards[-1])
        i = i + 1
        rew_median.append(np.median(cur_list))
    return rew_median

def smooth_max(rewards):
    rew_median = []
    i = 0
    while i < len(rewards) - MEDIAN_FC:
        cur_list = rewards[i: min(i + MEDIAN_FC, len(rewards))]
        while len(cur_list) < MEDIAN_FC:
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
    
    smoothed_rewards = smooth_fc(all_rewards)
    full_range = np.arange(0, len(all_rewards), 1)
    smooth_range = np.arange(0, len(smoothed_rewards), 1)

    print("Total number of steps:", sum(all_times))
    print('Average time length', np.mean(np.array(all_times)))
    plt.plot(full_range, all_rewards, '.', smooth_range, smoothed_rewards, '-')
    

def plot2(index1, index2, smooth_fc=smooth_median):
    with open(local_logs[index1], 'r') as csvfile:
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

    num_rows[row_index, 0] = num_rows[row_index, 0] + 1.2 ## change of initial coordinate

    all_rewards_1 = num_rows[row_index, 0].tolist()
    all_times_1 = num_rows[row_index, 1].tolist()
    #print("Total number of steps:", sum(all_times))
    #print('Average time length', np.mean(np.array(all_times)))
    sm_rewards_1 = smooth_fc(all_rewards_1)
    t_range_1 = np.arange(0, len(sm_rewards_1), 1)
    plt.plot(t_range_1, sm_rewards_1, '-')

    with open(local_logs[index2], 'r') as csvfile:
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
    all_rewards_2 = num_rows[row_index, 0].tolist()
    all_times_2 = num_rows[row_index, 1].tolist()
    sm_rewards_2 = smooth_fc(all_rewards_2)
    t_range_2 = np.arange(0, len(sm_rewards_2), 1)
    
    plt.plot(t_range_2, sm_rewards_2, '-')
    
    
def smooth(rewards, factor=10):
    smoothed_rew = []
    for i in range(0, len(rewards) - factor + 1):
        smoothed_rew.append(sum(rewards[i:i + factor]) / factor)
    return smoothed_rew

def main():
    plot(0, smooth_median)
    #plot(0, smooth_max)
    #plot2(0, 1, smooth_median)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smooth', type=int, default=1)
    args = parser.parse_args()
    MEDIAN_FC = args.smooth
    main()
