import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from realenv.core.render.profiler import Profiler


class FusePolicy(object):
    recurrent = False

    def __init__(self, name, ob_space, sensor_space, ac_space, save_per_acts, session, kind='large'):
        self.total_count = 0
        self.curr_count = 0
        self.save_per_acts = save_per_acts
        self.session = session
        with tf.variable_scope(name):
            self._init(ob_space, sensor_space, ac_space,  kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, sensor_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)
        assert isinstance(sensor_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        ob_sensor = U.get_placeholder(name="ob_sensor", dtype=tf.float32, shape=[sequence_length] + list(sensor_space.shape))

        y = ob_sensor
        y = tf.nn.relu(U.dense(y, 64, 'lin_ob', U.normc_initializer(1.0)))


        x = ob / 255.0
        if kind == 'small':  # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, 'lin', U.normc_initializer(1.0)))
        elif kind == 'large':  # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 64, 'lin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError


        x = tf.concat([x,y], 1)

        ## Saver
        # self.saver = tf.train.Saver()


        logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = U.dense(x, 1, "value", U.normc_initializer(1.0))[:, 0]

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
