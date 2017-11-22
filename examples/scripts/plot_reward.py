import numpy as np
import matplotlib.pyplot as plt
import time

def plot_reward(log_file, smooth_fc):
    episode_rewards = []
    total_timesteps = 0
    with open(log_file) as f:
        content = f.readlines()

    curr_episode = []
    for line in content:
        if "reward" in line:
            try:
                rew = float(line.split()[3])
                curr_episode.append(rew)
                total_timesteps = total_timesteps + 1
            except ValueError:
                continue
        if "Episode reset" in line:
            if not sum(curr_episode) < -500:
                episode_rewards.append(sum(curr_episode))
            curr_episode = []
    episode_rewards.append(sum(curr_episode))
    curr_episode = []
    print("Total episodes ", len(episode_rewards))
    print("Total timesteps", total_timesteps)
    #while True:
    t_range = np.arange(0, len(episode_rewards), 1)
    plt.plot(t_range, episode_rewards, '.', t_range, smooth_median(episode_rewards, smooth_fc), '-')
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
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', type=str, default="")
    parser.add_argument('--smooth', type=int, default=30)
    args = parser.parse_args()
    plot_reward(args.file, args.smooth)
