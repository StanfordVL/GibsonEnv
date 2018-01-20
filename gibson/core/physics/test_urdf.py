import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)

import pybullet as p
import pybullet_data



import time
cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
    p.connect(p.GUI)

p.resetSimulation()
p.setGravity(0, 0, -10)
p.loadSDF(os.path.join(pybullet_data.getDataPath(),"stadium.sdf"))
useRealTimeSim = 1

# for video recording (works best on Mac and Linux, not well on Windows)
# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
p.setRealTimeSimulation(useRealTimeSim)  # either this

object_ids = p.loadURDF(os.path.join(currentdir, 'models', 'husky.urdf'), flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

print(object_ids)
while (True):
    time.sleep(0.01)