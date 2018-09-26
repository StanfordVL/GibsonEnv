import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
import gym.spaces
import baselines.common.tf_util as U


## Fuse policy using PPO2 from OpenAI Baseline

class FusePolicy(object):
    def __init__(self, sess, ob_space, sensor_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        if isinstance(ac_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        ob_shape = (nbatch,) + ob_space.shape
        ob_sensor_shape = (nbatch,) + sensor_space.shape
        if self.is_discrete:
            actdim = ac_space.n
        else:
            actdim =  ac_space.shape[0]
        X_camera = tf.placeholder(tf.uint8, ob_shape, name='Ob_camera') #obs
        X_sensor = tf.placeholder(tf.float32, ob_sensor_shape, name='Ob_sensor')

        self.pdtype = make_pdtype(ac_space)

        with tf.variable_scope("model", reuse=reuse):
            h_camera = conv(tf.cast(X_camera, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2_camera = conv(h_camera, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3_camera = conv(h2_camera, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3_camera = conv_to_fc(h3_camera)
            h4_camera = fc(h3_camera, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi_camera = fc(h4_camera, 'pi', actdim, init_scale=0.01)
            vf_camera = fc(h4_camera, 'v', 1)[:,0]

        self.pd = self.pdtype.pdfromflat(pi_camera)

        with tf.variable_scope("model_sensor", reuse=reuse):
            h1_sensor = fc(X_sensor, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2_sensor = fc(h1_sensor, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi_sensor = fc(h2_sensor, 'pi', actdim, init_scale=0.01)
            h1_sensor = fc(X_sensor, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2_sensor = fc(h1_sensor, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf_sensor = fc(h2_sensor, 'vf', 1)[:,0]

        with tf.variable_scope("model", reuse=reuse):
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())
            X = tf.concat([X_camera, X_sensor], 0)
            pi_full = tf.concat([pi_camera, pi_sensor], 0)
            pi = fc(pi_full, 'pi', actdim, init_scale=0.01)
            vf_full = tf.concat([vf_camera, vf_sensor], 0)
            vf = fc(vf_full, 'vf', 1)[:,0]

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, ob_sensor, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X_camera:ob, X_sensor: ob_sensor})
            return a, v, self.initial_state, neglogp

        def value(ob, ob_sensor, *_args, **_kwargs):
            return sess.run(vf, {X_camera:ob, X_sensor: ob_sensor})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value



class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, is_discrete=False): #pylint: disable=W0613
        if isinstance(ac_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        print("nbatch%d" % (nbatch))

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        if self.is_discrete:
            nact = ac_space.n
        else:
            nact =  ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape) #obs
        
        with tf.variable_scope("model", reuse=reuse):
            h = conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            h5 = fc(h3, 'fc_vf', nh=512, init_scale=np.sqrt(2))


            pi = fc(h4, 'pi', nact, init_scale=0.05)
            vf = fc(h5, 'v', 1, act=lambda x: x)[:,0]

            if not self.is_discrete:
                logstd = tf.get_variable(name="logstd", shape=[1, nact],
                                     initializer=tf.zeros_initializer())

        self.pdtype = make_pdtype(ac_space)
        if self.is_discrete:
            self.pd = self.pdtype.pdfromflat(pi)
            a0 = self.pd.sample()
        else:
            pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
            self.pd = self.pdtype.pdfromflat(pdparam)
            a0 = self.pd.sample()

        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob.astype(np.float32)/255.0})
            assert(a.shape[0] == 1) # make sure a = a[0] don't throw away actions
            a = a[0]
            print(a,v, neglogp)
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})


        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        if isinstance(ac_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        ob_shape = (nbatch,) + ob_space.shape
        if self.is_discrete:
            actdim = ac_space.n
        else:
            actdim =  ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        if self.is_discrete:
            self.pd = self.pdtype.pdfromflat(pi)
            a0 = self.pd.sample()
        else:
            self.pd = self.pdtype.pdfromflat(pdparam)
            a0 = self.pd.sample()

        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            a = a[0]
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value



class MlpPolicy2(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        if isinstance(ac_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        ob_shape = (nbatch,) + ob_space.shape
        if self.is_discrete:
            actdim = ac_space.n
        else:
            actdim =  ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            X2 = conv_to_fc(X)
            h1 = fc(X2, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = fc(X2, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        if self.is_discrete:
            self.pd = self.pdtype.pdfromflat(pi)
            a0 = self.pd.sample()
        else:
            self.pd = self.pdtype.pdfromflat(pdparam)
            a0 = self.pd.sample()

        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob.astype(np.float32) / 255.0})
            a = a[0]
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy2(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        if isinstance(ac_space, gym.spaces.Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        ob_shape = (nbatch,) + ob_space.shape
        if self.is_discrete:
            actdim = ac_space.n
        else:
            actdim =  ac_space.shape[0]
        
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h_c = conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2_c = conv(h_c, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h2_c = conv_to_fc(h_c)
            h1 = fc(h2_c, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = fc(h2_c, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        if self.is_discrete:
            self.pd = self.pdtype.pdfromflat(pi)
            a0 = self.pd.sample()
        else:
            self.pd = self.pdtype.pdfromflat(pdparam)
            a0 = self.pd.sample()

        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob.astype(np.float32) / 255.0})
            a = a[0]
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

