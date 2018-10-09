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

    def get_motor_commands(self, amplitude_1, amplitude_2):
        self.t = self.step_count * self.time_step

        a1 = math.sin(self.t * self.period) * amplitude_1
        a2 = math.sin(self.t * self.period + math.pi) * amplitude_2
        a3 = math.sin(self.t * self.period) * amplitude_2
        a4 = math.sin(self.t * self.period + math.pi) * amplitude_1

        action = [a1, a2, a2, a1, a3, a4, a4, a3]

        self.step_count = self.step_count + 1

        return action

class ForwardSinePolicyController(SinePolicyController):
    def __init__(self, time_step=.001):
        SinePolicyController.__init__(self, time_step=time_step)

    def translate_action_to_motor_commands(self, a):
        return self.get_motor_commands(.7, .7)

class VectorSinePolicyController(SinePolicyController):
    def __init__(self, time_step=.001):
        SinePolicyController.__init__(self, time_step=time_step)

    def translate_action_to_motor_commands(self, a):
        phi, r = a
        # this constant translates +- pi/2 constraint to +- 3 steering amplitude
        c = -6. / math.pi
        steering_amplitude = c*phi
        amplitude_1 = max(r + steering_amplitude, .5)
        amplitude_2 = max(r - steering_amplitude, .5)
        return self.get_motor_commands(amplitude_1, amplitude_2)
