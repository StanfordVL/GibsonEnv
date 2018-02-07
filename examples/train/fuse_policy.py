import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from gibson.core.render.profiler import Profiler
import sys


class FusePolicy(object):
    recurrent = False

    def __init__(self, name, ob_space, sensor_space, ac_space, session, hid_size, num_hid_layers, save_per_acts=None, kind='large'):
        self.total_count = 0
        self.curr_count = 0
        self.save_per_acts = save_per_acts
        self.session = session
        with tf.variable_scope(name):
            self._init(ob_space, sensor_space, ac_space, hid_size, num_hid_layers, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, sensor_space, ac_space, hid_size, num_hid_layers, kind):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(sensor_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        ob_sensor = U.get_placeholder(name="ob_sensor", dtype=tf.float32, shape=[sequence_length] + list(sensor_space.shape))

        ## Obfilter on sensor output
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=sensor_space.shape)

        obz_sensor = tf.clip_by_value((ob_sensor - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        #x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))

        ## Adapted from mlp_policy
        last_out = obz_sensor
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="vffc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
        y = tf.layers.dense(last_out, 64, name="vffinal", kernel_initializer=U.normc_initializer(1.0))

        #y = ob_sensor
        #y = obz_sensor
        #y = tf.nn.relu(U.dense(y, 64, 'lin_ob', U.normc_initializer(1.0)))

        x = ob / 255.0
        if kind == 'small':  # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large':  # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 64, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        print(x.shape, y.shape)
        x = tf.concat([x,y], 1)

        ## Saver
        # self.saver = tf.train.Saver()


        logits = tf.layers.dense(x, pdtype.param_shape()[0], name="logits", kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name="value", kernel_initializer=U.normc_initializer(1.0))[:, 0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample()  # XXX
        self._act = U.function([stochastic, ob, ob_sensor], [ac, self.vpred, logits])


    def act(self, stochastic, ob, ob_sensor):
        ac1, vpred1, _ = self._act(stochastic, ob[None], ob_sensor[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1
        # if self.curr_count > self.save_per_acts:
        #    self.curr_count = self.curr_count - self.save_per_acts
        #    self.saver.save(self.session, 'cnn_policy',  global_step=self.total_count)
        return ac1[0], vpred1[0]


    def get_logits(self, stochastic, ob, ob_sensor):
        ac1, vpred1, logits = self._act(stochastic, ob[None], ob_sensor[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1
        # if self.curr_count > self.save_per_acts:
        #    self.curr_count = self.curr_count - self.save_per_acts
        #    self.saver.save(self.session, 'cnn_policy',  global_step=self.total_count)
        return ac1[0], vpred1[0], logits[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
