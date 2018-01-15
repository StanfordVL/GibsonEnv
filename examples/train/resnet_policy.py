from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype
from baselines import logger
import tensorflow as tf
import gym
from keras.preprocessing import image
from keras.applications import resnet50
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization

class ResnetPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, save_per_acts, session):
        self.total_count = 0
        self.curr_count = 0
        self.save_per_acts = save_per_acts
        self.session = session
        with tf.variable_scope(name):
            self._init(ob_space, ac_space)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        assert isinstance(ob_space, gym.spaces.Box)
        is_discrete = True
        if isinstance(ac_space, gym.spaces.Box):
            is_discrete = False

        # TODO: generalize this to continuous space

        self.obs_dim = len(ob_space.high)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        x = ob / 255.0
        input_shape = x.get_shape()[1:]
        logger.log("resnet policy: input shape: " + str(input_shape))

        weights = None
        if input_shape[-1] == 3:
            weights = 'imagenet'

        # build model here...
        resnet_model = resnet50.ResNet50(include_top=False, input_tensor=x, weights=weights)
        logger.log("resnet policy: obs dim: " + str(self.obs_dim))
        #logger.log("resnet policy: model output dim: " + str(resnet_model.output.get_shape()))

        x = U.flattenallbut0(x)

        pdparam = None
        if is_discrete:
            pdparam = U.dense(x, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        else:
            mean = U.dense(x, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)
        self.vpred = U.dense(x, 1, "value", U.normc_initializer(0.01))[:,0]

        ## Saver
        self.saver = tf.train.Saver()

        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        #ac = self.pd.sample()
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        self.total_count = self.total_count + 1
        self.curr_count = self.curr_count + 1
        if self.curr_count > self.save_per_acts:
            self.curr_count = self.curr_count - self.save_per_acts
            self.saver.save(self.session, 'husky_resnet_rgb_cont_policy', global_step=self.total_count)
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
