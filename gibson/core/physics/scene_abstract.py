import sys, os
sys.path.append(os.path.dirname(__file__))
import pybullet as p
import time
import gym


class Scene:
    "A base class for single- and multiplayer scenes"

    def __init__(self, gravity, timestep, frame_skip, env):
        #self.np_random, seed = gym.utils.seeding.np_random(None)
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.env = env

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(gravity, timestep, frame_skip)

        self.test_window_still_open = True  # or never opened
        self.human_render_detected = False  # if user wants render("human"), we open test window

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer: return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def episode_restart(self):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.clean_everything()
        #self.cpp_world.test_window_history_reset()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step(self.frame_skip)

class SingleRobotEmptyScene(Scene):
    multiplayer = False  # this class is used "as is" for InvertedPendulum, Reacher

class World:

    def __init__(self, gravity, timestep, frame_skip):
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.clean_everything()
        self._is_render = False
        self._last_frame_time = 0.0

    def clean_everything(self):
        #p.resetSimulation()
        p.setGravity(0, 0, -self.gravity)
        #p.setDefaultContactERP(0.9)
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=50, numSubSteps=(self.frame_skip-1))

    def step(self, frame_skip):
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            #time_to_sleep = self.timestep * self.frame_skip - time_spent
            #if time_to_sleep > 0:
            #    time.sleep(time_to_sleep)
        p.stepSimulation()


