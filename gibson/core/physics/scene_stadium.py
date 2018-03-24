from gibson.core.physics.scene_abstract import Scene
import pybullet as p
import os
import pybullet_data

class StadiumScene(Scene):
    zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen   = 105*0.25    # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50*0.25  # FOOBALL_FIELD_HALFWID

    def __init__(self, robot, gravity, timestep, frame_skip, env=None):
        Scene.__init__(self, gravity, timestep, frame_skip, env)

        filename = os.path.join(pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        for i in self.ground_plane_mjcf:
            pos, orn = p.getBasePositionAndOrientation(i)
            p.resetBasePositionAndOrientation(i, [pos[0], pos[1], pos[2] - 0.005], orn)

        for i in self.ground_plane_mjcf:
            p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.5])

        self.scene_obj_list = self.stadium


    def episode_restart(self):
        Scene.episode_restart(self)   # contains cpp_world.clean_everything()


class SinglePlayerStadiumScene(StadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False

class MultiplayerStadiumScene(StadiumScene):
    multiplayer = True
    players_count = 3
    def actor_introduce(self, robot):
        StadiumScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)

if __name__ == "__main__":

    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
        p.connect(p.GUI)

    scene = StadiumScene(gravity=9.8, timestep=0.0165, frame_skip=1)
    scene.episode_restart()
    while True:
        pass