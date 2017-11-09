import numpy as np
import matplotlib.pyplot as plt
import time

def plot_reward(log_file, smooth_fc):
    episode_rewards = []
    with open(log_file) as f:
        content = f.readlines()

    curr_episode = []
    for line in content:
        if "reward" in line:
            try:
                rew = float(line.split()[3])
                curr_episode.append(rew)
            except ValueError:
                continue
        if "Episode reset" in line:
            if not sum(curr_episode) < -500:
                episode_rewards.append(sum(curr_episode))
            curr_episode = []
    episode_rewards.append(sum(curr_episode))
    curr_episode = []
    print("Total episodes ", len(episode_rewards))
    print("Total timesteps", len(content))
    #while True:
    plt.plot(smooth(episode_rewards, smooth_fc))
    plt.show()

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
