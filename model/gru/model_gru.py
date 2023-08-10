import tensorflow as tf

# from Dice import dice
from rnn import dynamic_rnn
from utils import QAAttGRUCell, VecAttGRUCell


class Model(object):

    def __init__(self, item_count, cate_count, cate_list):
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])
        self.hist_i_neg = tf.placeholder(tf.int32, [None, None])  # [B, T]
        hidden_units = 128  # values

        item = tf.get_variable("item", [item_count, hidden_units // 2])
        cate = tf.get_variable("cate", [cate_count, hidden_units // 2])  # item_emb_w

        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))

        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        ic = tf.gather(cate_list, self.i)
        i_b = tf.gather(item_b, self.i)  # initialized by a constant, bias, wx+b

        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.i),
            tf.nn.embedding_lookup(cate, ic),
        ], axis=1)

        jc = tf.gather(cate_list, self.j)
        j_b = tf.gather(item_b, self.j)

        j_emb = tf.concat([
            tf.nn.embedding_lookup(item, self.j),
            tf.nn.embedding_lookup(cate, jc),
        ], axis=1)

        i_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.i),
            tf.nn.embedding_lookup(cate, ic),
        ], axis=1)

        j_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.j),
            tf.nn.embedding_lookup(cate, jc),
        ], axis=1)

        hc = tf.gather(cate_list, self.hist_i)

        h_t_k = tf.concat([
            tf.nn.embedding_lookup(item, self.hist_i),
            tf.nn.embedding_lookup(cate, hc),
        ], axis=2)

        rnn_output, hidden_state = dynamic_rnn(tf.nn.rnn_cell.GRUCell(hidden_units, reuse=tf.AUTO_REUSE), inputs=h_t_k,
                                               att_scores=None,
                                               sequence_length=tf.squeeze(self.sl,
                                                                          ), dtype=tf.float32, scope="gru1")

        rnn_output, hidden_state = dynamic_rnn(tf.nn.rnn_cell.GRUCell(hidden_units, reuse=tf.AUTO_REUSE), inputs=rnn_output,
                                               att_scores=None,
                                               sequence_length=tf.squeeze(self.sl,
                                                                          ), dtype=tf.float32, scope="gru2")

        hist_i = tf.layers.batch_normalization(inputs=hidden_state)
        hist_i = tf.reshape(hist_i, [-1, hidden_units])
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn_t')

        u_emb_i = hist_i
        u_emb_j = hist_i
        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())
        # -- fcn begin -------
        # gru_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        gru_i = tf.concat([u_emb_i, i_emb], axis=-1)
        gru_i = tf.layers.batch_normalization(inputs=gru_i, name='b1')
        d_layer_1_i = tf.layers.dense(gru_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        # gru_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        gru_j = tf.concat([u_emb_j, j_emb], axis=-1)
        gru_j = tf.layers.batch_normalization(inputs=gru_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(gru_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        self.logits = i_b + d_layer_3_i
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        print(self.p_and_n.get_shape().as_list())

        self.logits_i = i_b + d_layer_3_i
        self.logits_j = j_b + d_layer_3_j
        self.logits_i = tf.reshape(self.logits_i, [-1, 1])
        self.logits_j = tf.reshape(self.logits_j, [-1, 1])
        self.logits_i_j = tf.concat([self.logits_i, self.logits_j], axis=-1)
        y_p = tf.ones_like(self.logits_i)
        n_p = tf.zeros_like(self.logits_j)
        cross_p = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_i, labels=y_p))
        cross_n = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_j, labels=n_p))
        self.cross = cross_p + cross_n
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )
        # self.loss += aux_loss

        trainable_params = tf.trainable_variables()
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.opt = tf.train.AdamOptimizer()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run(
            [self.loss, self.train_op, ], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.y: uij[2],
                self.hist_i: uij[3],
                self.sl: uij[4],
                self.hist_i_neg: uij[5],
                self.lr: l,
            })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n, cross = sess.run([self.mf_auc, self.p_and_n, self.cross], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n, cross

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag=None):
    #:param h_states:
    #:param click_seq:
    #:param noclick_seq: #[B,T-1,E]
    #:param mask:#[B,1]
    #:param stag:
    #:return:
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    mask = tf.sequence_mask(tf.expand_dims(mask, 1), hist_len)
    mask = mask[:, 0, :]

    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)

    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    click_prop_ = auxiliary_net(click_input_, stag=stag)[:, :, 0]

    noclick_prop_ = auxiliary_net(noclick_input_, stag=stag)[
                    :, :, 0]  # [B,T-1]

    click_loss_ = - tf.reshape(tf.log(click_prop_),
                               [-1, tf.shape(click_seq)[1]]) * mask

    noclick_loss_ = - \
                        tf.reshape(tf.log(1.0 - noclick_prop_),
                                   [-1, tf.shape(noclick_seq)[1]]) * mask

    loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)

    return loss_


def auxiliary_net(in_, stag='auxiliary_net'):
    bn1 = tf.layers.batch_normalization(
        inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)

    dnn1 = tf.layers.dense(bn1, 100, activation=None,
                           name='f1' + stag, reuse=tf.AUTO_REUSE)

    dnn1 = tf.nn.sigmoid(dnn1)

    dnn2 = tf.layers.dense(dnn1, 50, activation=None,
                           name='f2' + stag, reuse=tf.AUTO_REUSE)

    dnn2 = tf.nn.sigmoid(dnn2)

    dnn3 = tf.layers.dense(dnn2, 1, activation=None,
                           name='f3' + stag, reuse=tf.AUTO_REUSE)

    y_hat = tf.nn.sigmoid(dnn3)

    return y_hat
