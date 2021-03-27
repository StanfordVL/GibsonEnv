import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
import pybullet_data

from gibson.data.datasets import get_model_path
from gibson.core.physics.scene_abstract import Scene
import pybullet as p


class BuildingScene(Scene):
    def __init__(self, robot, model_id, gravity, timestep, frame_skip, collision_enabled=True, env = None):
        Scene.__init__(self, gravity, timestep, frame_skip, env)

        # contains cpp_world.clean_everything()
        # stadium_pose = cpp_household.Pose()
        # if self.zero_at_running_strip_start_line:
        #    stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants
        
        filename = os.path.join(get_model_path(model_id), "mesh_z_up.obj")
        #filename = os.path.join(get_model_path(model_id), "3d", "blender.obj")
        #textureID = p.loadTexture(os.path.join(get_model_path(model_id), "3d", "rgb.mtl"))

        if robot.model_type == "MJCF":
            MJCF_SCALING = robot.mjcf_scaling
            scaling = [1.0/MJCF_SCALING, 1.0/MJCF_SCALING, 0.6/MJCF_SCALING]
        else:
            scaling  = [1, 1, 1]
        magnified = [2, 2, 2]

        if collision_enabled:
            collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)


            view_only_mesh = os.path.join(get_model_path(model_id), "mesh_view_only_z_up.obj")
            if os.path.exists(view_only_mesh):
                visualId = p.createVisualShape(p.GEOM_MESH,
                                       fileName=view_only_mesh,
                                       meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            else:
                visualId = -1

            boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
        else:
            visualId = p.createVisualShape(p.GEOM_MESH,
                                           fileName=filename,
                                           meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            boundaryUid = p.createMultiBody(baseCollisionShapeIndex = -1, baseVisualShapeIndex = visualId)

        p.changeDynamics(boundaryUid, -1, lateralFriction=1)
        #print(p.getDynamicsInfo(boundaryUid, -1))
        self.scene_obj_list = [(boundaryUid, -1)]       # baselink index -1
        

        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        
        if "z_offset" in self.env.config:
            z_offset = self.env.config["z_offset"]
        else:
            z_offset = -10 #with hole filling, we don't need ground plane to be the same height as actual floors

        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0], posObj = [0,0,z_offset], ornObj = [0,0,0,1])
        p.changeVisualShape(boundaryUid, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                            specularColor=[0.5, 0.5, 0.5])
        
    def episode_restart(self):
        Scene.episode_restart(self)


class SinglePlayerBuildingScene(BuildingScene):
    multiplayer = False
    def __init__(self, robot, model_id, gravity, timestep, frame_skip, collision_enabled=True, env = None):
        BuildingScene.__init__(self, robot=robot, model_id=model_id, gravity=gravity,
                               timestep=timestep, frame_skip=frame_skip, collision_enabled=collision_enabled, env=env)



