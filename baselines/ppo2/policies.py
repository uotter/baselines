import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, dot
from baselines.common.distributions import make_pdtype


def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def pure_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return h3


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnAttentionPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs

        actions_onehot = tf.eye(nact, batch_shape=[nbatch])
        batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, nact))
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            _input_dim = h.get_shape()[1].value
            batch_state_input_attention = tf.reshape(tf.tile(h, [1, nact]), shape=(-1, _input_dim))
            action_state_x = tf.concat([batch_state_input_attention, batch_actions_onehot], axis=1)
            state_attention_logits = fc(action_state_x, "attentions_output", _input_dim, init_scale=0.01)
            state_attention_prob = tf.nn.softmax(state_attention_logits, axis=1, name="state_attention_result_softmax")
            fc1 = tf.multiply(state_attention_prob, batch_state_input_attention, name="element_wise_weighted_states")
            pi = fc(fc1, 'batch_pi', nact, init_scale=0.01)
            pi = tf.reshape(tf.reduce_sum(tf.multiply(pi, batch_actions_onehot), axis=1), shape=(-1, nact), name='pi')
            vf = fc(h, 'v', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def get_attention(ob, *_args, **_kwargs):
            attention = sess.run([state_attention_prob], {X: ob})
            return attention

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.attention = state_attention_prob
        self.get_attention = get_attention


class MlpAttentionPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, sigmoid_attention=False, weak=False, deep=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        actions_onehot = tf.eye(actdim, batch_shape=[nbatch])
        batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, actdim))
        self.attention_size = X.get_shape()[1].value
        deep_attention = deep
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            _input_dim = X.get_shape()[1].value
            batch_state_input_attention = tf.reshape(tf.tile(X, [1, actdim]), shape=(-1, _input_dim))
            action_state_x = tf.concat([batch_state_input_attention, batch_actions_onehot], axis=1)
            if deep_attention:
                state_attention_logits = activ(fc(action_state_x, "attentions_output1", 128, init_scale=0.01))
                state_attention_logits = fc(state_attention_logits, "attentions_output", _input_dim, init_scale=0.01)
            else:
                state_attention_logits = fc(action_state_x, "attentions_output", _input_dim, init_scale=0.01)
            if sigmoid_attention:
                state_attention_prob = tf.nn.sigmoid(state_attention_logits, name="state_attention_result_sigmoid")
            else:
                state_attention_prob = tf.nn.softmax(state_attention_logits, axis=1, name="state_attention_result_softmax")
            self.attention_entropy_mean = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(state_attention_prob, 1e-10, 1.0)) * state_attention_prob, axis=1))
            self.attention_mean, self.attention_std = tf.nn.moments(state_attention_prob, name="soft_std", axes=[1])
            self.mean_attention_mean = tf.reduce_mean(self.attention_mean)
            self.mean_attention_std = tf.reduce_mean(self.attention_std)
            if weak:
                state_attention_prob_expand = state_attention_prob * self.attention_size
            else:
                state_attention_prob_expand = state_attention_prob
            fc1 = tf.multiply(state_attention_prob_expand, batch_state_input_attention, name="element_wise_weighted_states")
            fc1 = tf.concat([batch_state_input_attention,fc1], axis=1)
            h1 = activ(fc(fc1, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            pi = tf.reshape(tf.reduce_sum(tf.multiply(pi, batch_actions_onehot), axis=1), shape=(-1, actdim), name='pi')
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def get_attention(ob, *_args, **_kwargs):
            attention = sess.run(state_attention_prob, {X: ob})
            return attention

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.attention = state_attention_prob
        self.get_attention = get_attention


class MlpDotAttentionPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, sigmoid_attention=False, weak=False, deep=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        statedim = ob_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        actions_onehot_inline = tf.eye(actdim, batch_shape=[nbatch])
        batch_actions_onehot_inline = tf.reshape(actions_onehot_inline, shape=(-1, actdim))

        actions_onehot = tf.eye(actdim, batch_shape=[statedim])
        batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, actdim))
        batch_actions_onehot = tf.concat(values=tf.split(value=tf.reshape(batch_actions_onehot, shape=(-1, actdim * actdim)), num_or_size_splits=actdim, axis=1), axis=0)  # shape=(ns*na,na)
        batch_actions_onehot = tf.concat(values=tf.split(value=tf.tile(batch_actions_onehot, [1, nbatch]), num_or_size_splits=nbatch, axis=1), axis=0)  # shape=(ns*na,na)
        self.attention_size = X.get_shape()[1].value
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            _input_dim = X.get_shape()[1].value
            batch_state_input_attention = tf.reshape(tf.tile(X, [1, actdim]), shape=(-1, 1))  # shape=(ns*na,1)
            w = tf.get_variable("w", [actdim, 1], initializer=tf.random_uniform_initializer(-actdim, actdim))  # shape=(na,1)
            attention_a = tf.matmul(batch_actions_onehot, w, name="inside_matmul")  # shape=(ns*na,1)
            state_attention_logits = tf.multiply(attention_a, batch_state_input_attention, "attention_logits")  # shape=(ns*na,1)
            state_attention_logits = tf.reshape(state_attention_logits, shape=(-1, statedim))  # shape=(na,ns)
            print(state_attention_logits.get_shape().as_list())
            assert state_attention_logits.get_shape().as_list()[0] == nbatch * actdim, "state_attention_logits shape is not right."
            if sigmoid_attention:
                state_attention_prob = tf.nn.sigmoid(state_attention_logits, name="state_attention_result_sigmoid")
            else:
                state_attention_prob = tf.nn.softmax(state_attention_logits, axis=1, name="state_attention_result_softmax")
            self.attention_entropy_mean = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(state_attention_prob, 1e-10, 1.0)) * state_attention_prob, axis=1))
            self.attention_mean, self.attention_std = tf.nn.moments(state_attention_prob, name="soft_std", axes=[1])
            self.mean_attention_mean = tf.reduce_mean(self.attention_mean)
            self.mean_attention_std = tf.reduce_mean(self.attention_std)
            if weak:
                state_attention_prob_expand = state_attention_prob * self.attention_size
            else:
                state_attention_prob_expand = state_attention_prob
            batch_state_input_attention_inline = tf.reshape(tf.tile(X, [1, actdim]), shape=(-1, _input_dim))
            fc1 = tf.multiply(state_attention_prob_expand, batch_state_input_attention_inline, name="element_wise_weighted_states")
            h1 = activ(fc(fc1, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            pi = tf.reshape(tf.reduce_sum(tf.multiply(pi, batch_actions_onehot_inline), axis=1), shape=(-1, actdim), name='pi')
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def get_attention(ob, *_args, **_kwargs):
            attention = sess.run(state_attention_prob, {X: ob})
            return attention

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.attention = state_attention_prob
        self.get_attention = get_attention


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, sigmoid_attention=False, weak=False, deep=False):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
