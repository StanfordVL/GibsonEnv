from gibson.envs.husky_env import HuskyNavigateEnv
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'test_control.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = HuskyNavigateEnv(config = args.config)
    env.reset()

    base_action = np.array([-0.001,-0.001,-0.001,-0.001])
    control_signal = 1
    v = 0
    kp = 50
    ki = 0.1
    kd = 15
    ie = 0
    de = 0
    olde = 0
    for i in range(1000):
        obs, _, _, _ = env.step([0,0,0,0])
    vs = []
    control_signals = []
    for i in range(6000):
        e = control_signal - v
        de = e - olde
        ie += e
        olde = e
        pid = kp * e + ki * ie + kd * de
        obs, _, _, _ = env.step(action = pid * base_action)
        v = np.linalg.norm(obs["nonviz_sensor"][-3:])
        vs.append(v)
        control_signals.append(control_signal)

    control_signal = 0.5
    for i in range(6000):
        e = control_signal - v
        de = e - olde
        ie += e
        olde = e
        pid = kp * e + ki * ie + kd * de
        obs, _, _, _ = env.step(action = pid * base_action)
        v = np.linalg.norm(obs["nonviz_sensor"][-3:])
        vs.append(v)
        control_signals.append(control_signal)

    control_signal = 1.5
    for i in range(6000):
        e = control_signal - v
        de = e - olde
        ie += e
        olde = e
        pid = kp * e + ki * ie + kd * de
        obs, _, _, _ = env.step(action = pid * base_action)
        v = np.linalg.norm(obs["nonviz_sensor"][-3:])
        vs.append(v)
        control_signals.append(control_signal)

    plt.plot(vs)
    plt.plot(control_signals)
    plt.legend(["speed", "target speed"])
    plt.savefig("pid.pdf")