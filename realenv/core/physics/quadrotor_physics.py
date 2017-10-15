from render_physics import PhysRenderer
from PhysicsObject import PhysicsObject
import pybullet as p
import settings
import numpy as np
import time

class QuadObject(PhysicsObject):
    def __init__(self, uid, simulator, pos, quat, v_t, v_r, v_f, fps):
        super(self.__class__, self).__init__(uid, simulator, pos, quat, v_t, v_r, fps)
        self.v_f = v_f
        l_prop = .1750
        self.props = [[0, l_prop, 0], [0, -l_prop, 0], [l_prop, 0, 0], [-l_prop, 0, 0]]
        return

    def getUpdateFromKeyboard(self, restart):
        keys = super(self.__class__, self).getUpdateFromKeyboard()    
        if (ord('1') in keys or restart):
            self.action['rotor_1'] = True
        if (ord('2') in keys):
            self.action['rotor_2'] = True
        if (ord('3') in keys):
            self.action['rotor_3'] = True
        if (ord('4') in keys):
            self.action['rotor_4'] = True

    def parseActionAndUpdate(self, action=None):
        """ Update position: because the object's rotation
        changes every time, the position needs to be updated
        by delta
        """
        super(self.__class__, self).parseActionAndUpdate(action)
        
        delta_force = np.array([0, 0, 0, 0], dtype=float)
        if self.action['rotor_1']:
            delta_force[0] = self.v_f/settings.STEPS_PER_SEC
        if self.action['rotor_2']:
            delta_force[1] = self.v_f/settings.STEPS_PER_SEC
        if self.action['rotor_3']:
            delta_force[2] = self.v_f/settings.STEPS_PER_SEC
        if self.action['rotor_4']:
            delta_force[3] = self.v_f/settings.STEPS_PER_SEC
        self.d_force = delta_force

        self.updateAppliedForces()
        #self._clearUpDelta()

    def updateAppliedForces(self):
        super(self.__class__, self).updatePositionOrientation()
        rotor_keys = ['rotor_1', 'rotor_2', 'rotor_3', 'rotor_4']
        for i, force in enumerate(self.d_force):
            posObj = self.props[i]
            forceObj = [0, 0, force]
            print(posObj, forceObj)
            p.applyExternalForce(self.uid, -1, forceObj, posObj, p.LINK_FRAME)

    def clearUpDelta(self):
        super(self.__class__, self).clearUpDelta()
        self.d_force = np.array([0, 0, 0, 0], dtype=float)

    def _createDefaultAction(self):
        action = super(self.__class__, self)._createDefaultAction()
        action['rotor_1'] = False
        action['rotor_2'] = False
        action['rotor_3'] = False
        action['rotor_4'] = False
        return action


class QuadRenderer(PhysRenderer):
    def initialize(self, pose):
        pos, quat_xyzw = pose[0], pose[1]
        v_t = 1             # 1m/s max speed
        v_r = np.pi/5       # 36 degrees/s
        self.cart = QuadObject(self.objectUid, p, pos, quat_xyzw, v_t, v_r, 2500.0,  self.framePerSec)
        print("Generated cart", self.objectUid)
        #p.setTimeStep(1.0/framePerSec)
        p.setTimeStep(1.0/settings.STEPS_PER_SEC)


if __name__ == "__main__":
    datapath = "../data"
    model_id = "11HB6XZSh1Q"

    framePerSec = 13

    pose_init = ([-5.767663955688477, -5.19164514541626, 1.5544220209121704], [0.2869106641737875, 0.2873109019524023, 0.6437986789718354, 0.6485815117290755])

    r_physics = QuadRenderer(datapath, model_id, framePerSec, debug=True,human=True)
    r_physics.initialize(pose_init)
    while True:
        r_physics.renderToScreen(action = None)
        '''
        {
        'rotor_1': True,
        'rotor_2': True,
        'rotor_3': True,
        'rotor_4': True})'''
        time.sleep(0.01)