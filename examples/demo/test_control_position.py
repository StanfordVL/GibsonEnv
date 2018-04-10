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

    base_action_omage = np.array([-0.001,0.001,-0.001,0.001])
    base_action_v = np.array([0.001, 0.001, 0.001, 0.001])

    control_signal = -0.5
    control_signal_omega = 0.5
    v = 0
    omega = 0
    kp = 100
    ki = 0.1
    kd = 25
    ie = 0
    de = 0
    olde = 0
    ie_omega = 0
    de_omega = 0
    olde_omage = 0
    for i in range(1000):
        obs, _, _, _ = env.step([0,0,0,0])
    vs = []
    control_signals = []
    omegas = []
    control_signals_omega = []
    position = []

    env.robot.target_pos = env.robot.body_xyz + np.array([20,-5,0])

    for i in range(32000):
        control_signal = min(env.robot.dist_to_target() * 0.2, 1)
        if (env.robot.dist_to_target() < 0.1):
            break
        control_signal_omega = env.robot.angle_to_target
        print(env.robot.dist_to_target())
        e = control_signal - v
        de = e - olde
        ie += e
        olde = e
        pid_v = kp * e + ki * ie + kd * de

        e_omega = control_signal_omega - omega
        de_omega = e_omega - olde_omage
        ie_omega += e_omega
        pid_omega = kp * e_omega + ki * ie_omega + kd * de_omega

        obs, _, _, _ = env.step(action=pid_v * base_action_v + pid_omega * base_action_omage)
        v = obs["nonviz_sensor"][3]
        omega = obs["nonviz_sensor"][-1]

        #print(env.robot.dist_to_target(), env.robot.angle_to_target)

        vs.append(v)
        control_signals.append(control_signal)
        omegas.append(omega)
        control_signals_omega.append(control_signal_omega)

        position.append(env.robot.body_xyz)

    plt.figure()
    plt.plot(vs, alpha=.7)
    plt.plot(control_signals, alpha=.7)
    plt.plot(omegas, alpha=.7)
    plt.plot(control_signals_omega, alpha=.7)

    plt.legend(["speed", "target speed", "angular speed", "target angular speed"])
    plt.savefig("position_pid.pdf")

    plt.figure()
    plt.plot([item[0] for item in position], [item[1] for item in position])
    plt.plot(env.robot.target_pos[0], env.robot.target_pos[1], '*')
    plt.axis('equal')
    plt.savefig("traj.pdf")