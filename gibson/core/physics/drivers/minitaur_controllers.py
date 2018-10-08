import numpy as np
import math

class SinePolicyController:
    """Have minitaur walk with a sine gait."""
    # Leg model enabled
    # Accurate model enabled
    # pd control enabled
    def __init__(self, time_step=.001):
        self.period = 40
        self.step_count = 0
        self.time_step = time_step
        self.t = 0

    def translate_action_to_motor_commands(self, a):
        self.t = self.step_count * self.time_step
        actions = self.get_motor_commands()
        self.step_count = self.step_count + 1
        return actions

    def get_motor_commands(self):
        w1 = .7
        w2 = .7

        a1 = math.sin(self.t * self.period) * w1
        a2 = math.sin(self.t * self.period + math.pi) * w2
        a3 = math.sin(self.t * self.period) * w2
        a4 = math.sin(self.t * self.period + math.pi) * w1

        action = [a1, a2, a2, a1, a3, a4, a4, a3]

        return action

class VelocitySinePolicyController:
    """Have minitaur walk with a sine gait."""
    # Leg model enabled
    # Accurate model enabled
    # pd control enabled
    def __init__(self):
        self.period = 60
        self.step_count = 0
        self.time_step = 0.001
        self.t = 0

    def translate_action_to_motor_commands(self, velocity_vec):
        self.t = self.step_count * self.time_step
        phi, r = velocity_vec
        # this constant translates +- pi/2 constraint to +- 3 steering amplitude
        c = -6. / math.pi
        steering_amplitude = c*phi
        actions = self.get_motor_commands(r, steering_amplitude)
        self.step_count = self.step_count + 1
        return actions

    def get_motor_commands(self, r, steering_amplitude):
        # Applying asymmetrical sine gaits to different legs can steer the minitaur
        # Legs need minimum amplitude to move
        w1 = max(r + steering_amplitude, .5)
        w2 = max(r - steering_amplitude, .5)

        a1 = math.sin(self.t * self.period) * w1
        a2 = math.sin(self.t * self.period + math.pi) * w2
        a3 = math.sin(self.t * self.period) * w2
        a4 = math.sin(self.t * self.period + math.pi) * w1

        action = [a1, a2, a2, a1, a3, a4, a4, a3]
        return action
