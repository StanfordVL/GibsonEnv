import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
import pybullet_data

from realenv import configs
from realenv.data.datasets import get_model_path
from realenv.core.physics.scene_abstract import Scene
import pybullet as p


class BuildingScene(Scene):
    def __init__(self, robot, model_id, gravity, timestep, frame_skip):
        Scene.__init__(self, gravity, timestep, frame_skip)   

        # contains cpp_world.clean_everything()
        # stadium_pose = cpp_household.Pose()
        # if self.zero_at_running_strip_start_line:
        #    stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants
        
        #filename = os.path.join(get_model_path(configs.NAVIGATE_MODEL_ID), "modeldata", "out_z_up.obj")
        #filename = os.path.join(get_model_path(configs.NAVIGATE_MODEL_ID), "modeldata", "out_z_up.obj")
        #filename = os.path.join(get_model_path(model_id), "modeldata", "out_z_up_smoothed.obj")
        filename = os.path.join(get_model_path(model_id), "modeldata", "out_z_up.obj")
        #filename = os.path.join(get_model_path(model_id), "3d", "rgb.obj")
        #filename = os.path.join(get_model_path(model_id), "3d", "blender.obj")
        #textureID = p.loadTexture(os.path.join(get_model_path(model_id), "3d", "rgb.mtl"))

        if robot.model_type == "MJCF":
            MJCF_SCALING = robot.mjcf_scaling
            scaling = [1.0/MJCF_SCALING, 1.0/MJCF_SCALING, 0.6/MJCF_SCALING]
        else:
            scaling  = [1, 1, 1]
        magnified = [2, 2, 2]

        collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        #print("collision id", collisionId, "textureID", textureID)

        #textureID = p.createVisualShape(p.GEOM_MESH, fileName=filename)
        textureID = -1

        #p.changeVisualShape(collisionId, -1, textureUniqueId= textureID)

        #visualId = p.createVisualShape(p.GEOM_MESH, fileName=filename, meshScale=original, rgbaColor = [93/255.0,95/255.0, 96/255.0,0.75], specularColor=[0.4, 0.4, 0.4])
        boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = textureID)
        #visualId = p.loadTexture(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tex256.png"))
        #p.changeVisualShape(boundaryUid, -1, textureUniqueId=visualId)
        #self.scene_obj = [collisionId]
        #planeName = os.path.join(pybullet_data.getDataPath(),"mjcf/ground_plane.xml")
        #self.ground_plane_mjcf = p.loadMJCF(planeName)
        #print("built plane", type(self.ground_plane_mjcf))
        p.changeDynamics(boundaryUid, -1, lateralFriction=0.6, spinningFriction=0.1, rollingFriction=0.1)
        self.scene_obj_list = [boundaryUid]
        #self.scene_obj = (int(p.loadURDF(filename)), )


        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        #print(self.ground_plane_mjcf)

        if model_id in configs.OFFSET_GROUND.keys():
            z_offset = configs.OFFSET_GROUND[model_id]
        else:
            z_offset = -0.5

        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0], posObj = [0,0,z_offset], ornObj = [0,0,0,1])
        #collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        #p.changeVisualShape(boundaryUid, -1, textureUniqueId=visualId)
        #p.changeVisualShape(i,-1,rgbaColor=[93/255.0,95/255.0, 96/255.0,0.75], specularColor=[0.4, 0.4, 0.4])

        #p.changeVisualShape(i,-1,rgbaColor=[93/255.0,95/255.0, 96/255.0,0.55], specularColor=[0.4, 0.4, 0.4])
        #p.changeVisualShape(boundaryUid,-1,rgbaColor=[198/255.0,183/255.0, 115/255.0, 1.0], specularColor=[0.5, 0.5, 0.5])
        #p.changeVisualShape(self.scene_obj,-1,rgbaColor=[198/255.0,183/255.0, 115/255.0, 1.0], specularColor=[1, 1, 1])
    
    def episode_restart(self):
        Scene.episode_restart(self)


class SinglePlayerBuildingScene(BuildingScene):
    multiplayer = False
    def __init__(self, robot, model_id, gravity, timestep, frame_skip):
        BuildingScene.__init__(self, robot, model_id, gravity, timestep, frame_skip)



