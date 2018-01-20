import numpy as np
import gibson.core.physics.settings as settings
from transforms3d import euler, quaternions


class PhysicsObject(object):
    """
    Controllable object in world
    This class mostly handles action parsing

    Note: only stores realtime delta pose, absolute pose is not
    stored

    Note: conventions
    By OpenGL convention, object is default: +x facing,
    which is incompatible with camera view matrix (-z facing).
    The inconsistency is handled by _cameraCalibrate function.
    By default, internal quaternion variables are [x, y, z, w]
    """

    def __init__(self, uid, simulator, pos, quat, v_t, v_r, fps):
        self.uid = uid
        self.sim = simulator

        self.v_t = float(v_t)
        self.v_r = float(v_r)
        self.fps = float(fps)
        self.action = self._createDefaultAction()

        #self.camera_offset  = np.array([0, 0, 0.8])
        self.camera_offset  = np.array([0, 0, 0])

        self.pos_init_xyz   = np.array(pos)
        self.quat_init_xyzw = np.array(quat)

        ## Relative delta rotation of object to world
        self.d_alpha, self.d_beta, self.d_gamma = 0, 0, 0

        ## Relative delta position inside object's world view
        self.d_xyz = np.array([0, 0, 0], dtype=float)
        self._updateInitialPositionOrientation()

        ## DEPRECATED: roll, pitch, yaw
        self.roll, self.pitch, self.yaw = 0, np.pi/6, 0


    def _updateInitialPositionOrientation(self):
        """ Update initial physics simulation
        Similar to implementation in updatePositionOrientation()
        """
        pos_world_xyz   = self.pos_init_xyz
        quat_world_xyzw = self.quat_init_xyzw

        ## parse xyz, alpha, beta, gamma from world
        ## apply local delta xyz, delta alpha, delta beta, delta gamma
        ## update to world
        quat_world_xyzw = self._rotateIntrinsic(quat_world_xyzw)
        ## Calibrate
        quat_world_xyzw = self._cameraCalibrate(quat_world_xyzw)
        #pos_xyz   = self._translateIntrinsic()
        self.sim.resetBasePositionAndOrientation(self.uid, pos_world_xyz, quat_world_xyzw)


    def updatePositionOrientation(self):
        """ Update physics simulation
        Physics simulation = object position + object rotation

        Note: When using sim.stepSimulation(), you need to call
        updatePositionOrientation() periodically.
        """
        pos_world_xyz, quat_world_xyzw = self.sim.getBasePositionAndOrientation(self.uid)
        # velocity
        v_translate, v_rotate = self.sim.getBaseVelocity(self.uid)

        quat_world_xyzw     = self._cameraUncalibrate(quat_world_xyzw)

        new_pos_world_xyz   = self._translateIntrinsic(pos_world_xyz, quat_world_xyzw)
        new_quat_world_xyzw = self._rotateIntrinsic(quat_world_xyzw)
        ## Calibrate
        new_quat_world_xyzw = self._cameraCalibrate(new_quat_world_xyzw)

        self.sim.resetBasePositionAndOrientation(self.uid, new_pos_world_xyz, new_quat_world_xyzw)
        self.sim.resetBaseVelocity(self.uid, v_translate, v_rotate)


    def getViewPosAndOrientation(self):
        """Output real-time pose for view renderer

        Note: the output pose & orientation are absolute pose defined
        inside physics world coordinate. The viewer might use different
        convention, e.g. a coordinate relative to initial pose.
        PhysicsObject is agnostic to viewer convention. It is the viewer's
        job to handle this input/output
        """
        pos_world_xyz, quat_world_xyzw = self.sim.getBasePositionAndOrientation(self.uid)

        pos_world_xyz         = np.array(pos_world_xyz)
        quat_world_xyzw     = np.array(quat_world_xyzw)

        quat_world_xyzw     = self._cameraUncalibrate(quat_world_xyzw)
        quat_world_wxyz     = PhysicsObject.quatXyzwToWxyz(quat_world_xyzw)

        """
        quat_init_wxyz     = PhysicsObject.quatXyzwToWxyz(self.quat_init_xyzw)
        pos_view_xyz     = pos_world_xyz - self.pos_init_xyz
        quat_view_wxyz     = quaternions.qmult(quaternions.qinverse(quat_init_wxyz), quat_world_wxyz)
        euler_view       = euler.quat2euler(quat_view_wxyz)
        """

        pos_world_xyz = self._applyCameraOffset(pos_world_xyz)

        return pos_world_xyz.tolist(), quat_world_wxyz.tolist()



    def getUpdateFromKeyboard(self, restart=False):
        # Special Controls: B3G_RIGHT_ARROW, B3G_LEFT_ARROW,
        #     B3G_DOWN_ARROW, B3G_UP_ARROW

        self.action = self._createDefaultAction()
        keys = self.sim.getKeyboardEvents()


        if (ord('r') in keys or restart):
            self.action['restart'] = True
        if (ord('d') in keys):
            self.action['right'] = True
        if (ord('a') in keys):
            self.action['left'] = True
        if (ord('s') in keys):
            self.action['backward'] = True
        if (ord('w') in keys or ord('q') in keys):
            self.action['forward'] = True
        if (ord('z') in keys):
            self.action['up'] = True
        if (ord('c') in keys):
            self.action['down'] = True

        if (ord('u') in keys):
            self.action['alpha'] = 1
        if (ord('j') in keys):
            self.action['alpha'] = -1
        if (ord('i') in keys):
            self.action['beta'] = 1
        if (ord('k') in keys):
            self.action['beta'] = -1
        if (ord('o') in keys):
            self.action['gamma'] = 1
        if (ord('l') in keys):
            self.action['gamma'] = -1
        return keys
        #self.parseActionAndUpdate()


    def parseActionAndUpdate(self, action=None):
        """ Update position: because the object's rotation
        changes every time, the position needs to be updated
        by delta
        """
        if action:
            self.action = self._createDefaultAction()
            for k in action.keys():
                self.action[k] = action[k]
        delta_xyz = np.array([0, 0, 0], dtype=float)
        if self.action['restart']:
            self._restartLocationOrientation()
            return
        if self.action['up']:
            delta_xyz[1] =  self.v_t/settings.STEPS_PER_SEC
        if self.action['down']:
            delta_xyz[1] = -self.v_t/settings.STEPS_PER_SEC
        if self.action['left']:
            delta_xyz[0] = -self.v_t/settings.STEPS_PER_SEC
        if self.action['right']:
            delta_xyz[0] =  self.v_t/settings.STEPS_PER_SEC
        if self.action['forward']:
            delta_xyz[2] = -self.v_t/settings.STEPS_PER_SEC
        if self.action['backward']:
            delta_xyz[2] =  self.v_t/settings.STEPS_PER_SEC
        self.d_xyz = delta_xyz

        ## Update rotation: reset the rotation every time
        if self.action['alpha'] > 0:
            self.d_alpha =  self.v_r/settings.STEPS_PER_SEC
        if self.action['alpha'] < 0:
            self.d_alpha = -self.v_r/settings.STEPS_PER_SEC
        if self.action['beta'] > 0:
            self.d_beta =   self.v_r/settings.STEPS_PER_SEC
        if self.action['beta'] < 0:
            self.d_beta =  -self.v_r/settings.STEPS_PER_SEC
        if self.action['gamma'] > 0:
            self.d_gamma =  self.v_r/settings.STEPS_PER_SEC
        if self.action['gamma'] < 0:
            self.d_gamma = -self.v_r/settings.STEPS_PER_SEC

        self.updatePositionOrientation()
        #self.clearUpDelta()

    @staticmethod
    def quatWxyzToXyzw(wxyz):
        """
        wxyz: transforms3s array format
        xyzw: pybullet format
        """
        return np.concatenate((wxyz[1:], wxyz[:1]))

    @staticmethod
    def quatXyzwToWxyz(xyzw):
        """
        wxyz: transforms3s array format
        xyzw: pybullet format
        """
        return np.concatenate((xyzw[-1:], xyzw[:-1]))


    def rotate_quat_by_euler(xyzw, e_x, e_y, e_z):
        """
        wxyz: transforms3s array format
        xyzw: pybullet format
        """
        wxyz = PhysicsObject.quatXyzwToWxyz(xyzw)
        rot_mat = euler.euler2mat(e_x, e_y, e_z)
        wxyz = quaternions.qmult(rot_mat, wxyz)
        return PhysicsObject.quatWxyzToXyzw(wxyz)


    def _applyCameraOffset(self, pos_xyz):
        pos_camera_xyz = pos_xyz + self.camera_offset
        return pos_camera_xyz


    def _cameraCalibrate(self, org_quat_xyzw):
        """ Convert object's head from +x facing to -z facing
        Note that this is not needed when you're computing view_matrix,
        only use this function for adjusting object head
        To get object rotation at current object pose:
            rotation = self._cameraCalibrate(self._rotateIntrinsic(
                   self.quat_world))
        To get view rotation at current object pose:
              rotation = self._rotateIntrinsic(self.quat_world)
          """
        z_facing_wxyz = euler.euler2quat(-np.pi/2, np.pi/2, 0)
        org_quat_wxyz = PhysicsObject.quatXyzwToWxyz(org_quat_xyzw)
        new_quat_xyzw = PhysicsObject.quatWxyzToXyzw(quaternions.qmult(org_quat_wxyz, z_facing_wxyz))
        return new_quat_xyzw


    def _createDefaultAction(self):
        action = {
            'up'      : False,
            'down'      : False,
            'left'      : False,
            'right'      : False,
            'forward' : False,
            'backward': False,
            'restart' : False,
            'alpha'   : 0,
            'beta'    : 0,
            'gamma'   : 0
        }
        return action

    def _cameraUncalibrate(self, new_quat_xyzw):
        """ Undo the effect of _cameraCalibrate
        Needed in two cases:
        (1) Updating pose in physics engine

            self._cameraUnalibrate(sim.getBasePositionAndOrientation(uid))

        This is because object orientation in physics engine has
        been calibrated. To update parameter (alpha, beta, gamma),
        need to uncalibrate first. See updatePositionOrientation()
          (2) Sending pose to viewer renderer

             send(self._cameraUnalibrate(sim.See updatePositionOrientation(uid)))

        This is because we need to send uncalibrated view pose
        """
        x_facing_wxyz = euler.euler2quat(np.pi/2, 0, -np.pi/2)
        new_quat_wxyz = PhysicsObject.quatXyzwToWxyz(new_quat_xyzw)
        org_quat_wxyz = PhysicsObject.quatWxyzToXyzw(quaternions.qmult(new_quat_wxyz, x_facing_wxyz))
        return org_quat_wxyz


    def _translateIntrinsic(self, pos_world_xyz, quat_world_xyzw):
        """ Add intrinsic translation to extrinsic
        Extrinsic translation = Intrinsic delta translation +
            Extrinsic translation

        Note: order doesn't matter
        """
        delta_objec_xyz = self.d_xyz
        quat_world_wxyz = PhysicsObject.quatXyzwToWxyz(quat_world_xyzw)
        delta_world_xyz = quaternions.quat2mat(quat_world_wxyz).dot(delta_objec_xyz)
        return pos_world_xyz + delta_world_xyz


    def _rotateIntrinsic(self, quat_world_xyzw):
        """ Add intrinsic rotation to extrinsic
        Extrinsic rotation = Extrinsic rotation *
            Intrinsic delta rotation

        Note: order matters. Apply intrinsic delta quat first, then
        extrinsic rotation
        """
        quat_world_wxyz = PhysicsObject.quatXyzwToWxyz(quat_world_xyzw)
        euler_world     = euler.quat2euler(quat_world_wxyz)

        quat_objec_wxyz = euler.euler2quat(self.d_alpha, self.d_beta, self.d_gamma)
        new_quat_world_wxyz  = quaternions.qmult(quat_world_wxyz, quat_objec_wxyz)
        new_quat_world_xyzw  = PhysicsObject.quatWxyzToXyzw(new_quat_world_wxyz)
        return new_quat_world_xyzw


    def _restartLocationOrientation(self):
        self._updateInitialPositionOrientation()


    def clearUpDelta(self):
        self.d_xyz = np.array([0, 0, 0], dtype=float)
        self.d_alpha, self.d_beta, self.d_gamma = 0, 0, 0


    ## DEPRECATED: roll, pitch, yaw
    def _principle_to_mat(self):
        alpha = self.yaw
        beta  = self.pitch
        gamma = self.roll
        mat   = np.eye(4)
        mat[0, 0] = cos(alpha)*cos(beta)
        mat[1, 0] = sin(alpha)*cos(beta)
        mat[2, 0] = -sin(beta)

        mat[0, 1] = cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)
        mat[1, 1] = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
        mat[2, 1] = cos(beta)*sin(gamma)

        mat[0, 2] = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
        mat[1, 2] = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)
        mat[2, 2] = cos(beta)*cos(gamma)
        return mat
