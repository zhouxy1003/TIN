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
        # hidden_units_copy = 128  # keys, queries

        # position = tf.get_variable("pos", shape=[431, hidden_units])  # time embedding 31 buckets


        # self.p = position

        item_target_v = tf.get_variable("item_t_v", [item_count, hidden_units // 2])  # item_emb_w
        item_target_k = tf.get_variable("item_t_k", [item_count, hidden_units // 2])
        item_target_q = tf.get_variable("item_t_q", [item_count, hidden_units // 2])

        cate_target_v = tf.get_variable("cate_t_v", [cate_count, hidden_units // 2])  # item_emb_w
        cate_target_k = tf.get_variable("cate_t_k", [cate_count, hidden_units // 2])
        cate_target_q = tf.get_variable("cate_t_q", [cate_count, hidden_units // 2])

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
        # i_pos_emb = tf.nn.embedding_lookup(position, self.pos)
        # i_emb = tf.add(i_emb, i_pos_emb)

        jc = tf.gather(cate_list, self.j)
        j_b = tf.gather(item_b, self.j)

        j_emb = tf.concat([
            tf.nn.embedding_lookup(item, self.j),
            tf.nn.embedding_lookup(cate, jc),
        ], axis=1)
        # j_pos_emb = i_pos_emb
        # j_emb = tf.add(j_emb, j_pos_emb)

        i_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.i),  # item_target_q
            tf.nn.embedding_lookup(cate, ic),  # cate_target_q
        ], axis=1)
        # i_pos_tar = tf.nn.embedding_lookup(position, self.pos)  # position_target
        # i_t_q = tf.add(i_t_q, i_pos_tar)

        j_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.j),  # item_target_q
            tf.nn.embedding_lookup(cate, jc),  # cate_target_q
        ], axis=1)
        # j_pos_tar = i_pos_tar
        # j_t_q = tf.add(j_t_q, j_pos_tar)

        hc = tf.gather(cate_list, self.hist_i)
        hc_neg = tf.gather(cate_list, self.hist_i_neg)

        h_t_v = tf.concat([
            tf.nn.embedding_lookup(item, self.hist_i),
            tf.nn.embedding_lookup(cate, hc),
        ], axis=2)
        # h_pos = tf.nn.embedding_lookup(position, self.pos_i)
        # h_t_v = tf.add(h_t_v, h_pos)

        h_t_k = tf.concat([
            tf.nn.embedding_lookup(item, self.hist_i),  # item_target_q
            tf.nn.embedding_lookup(cate, hc),  # cate_target_q
        ], axis=2)

        h_t_k_neg = tf.concat([
            tf.nn.embedding_lookup(item, self.hist_i_neg),  # item_target_q
            tf.nn.embedding_lookup(cate, hc_neg),  # cate_target_q
        ], axis=2)
        # h_pos_tar = tf.nn.embedding_lookup(position, self.pos_i)  # position_target
        # h_t_k = tf.add(h_t_k, h_pos_tar)
        rnn_output, hidden_state = dynamic_rnn(tf.nn.rnn_cell.GRUCell(hidden_units, reuse=tf.AUTO_REUSE), inputs=h_t_k,
                                               att_scores=None,
                                               sequence_length=tf.squeeze(self.sl,
                                                                          ), dtype=tf.float32, scope="gru1")

        aux_loss = auxiliary_loss(rnn_output[:, :-1, :], h_t_k[:, 1:, :], h_t_k_neg[:, 1:, :], tf.subtract(self.sl, 1), stag="gru")

        scores_i_t = target_attention(i_t_q, rnn_output, h_t_v, self.sl)
        rnn_output_i, hidden_state_i = dynamic_rnn(VecAttGRUCell(hidden_units), inputs=rnn_output,
                                                   att_scores=tf.transpose(scores_i_t, perm=[0, 2, 1]),
                                                   sequence_length=tf.squeeze(self.sl,
                                                                              ), dtype=tf.float32, scope="gru2")
        # hist_i_t = tf.expand_dims(hidden_state_i, axis=1)
        hist_i_t = tf.layers.batch_normalization(inputs=hidden_state_i)
        hist_i_t = tf.reshape(hist_i_t, [-1, hidden_units])
        hist_i_t = tf.layers.dense(hist_i_t, hidden_units, name='hist_fcn_t')

        scores_j_t = target_attention(j_t_q, rnn_output, h_t_v, self.sl)
        rnn_output_j, hidden_state_j = dynamic_rnn(VecAttGRUCell(hidden_units, reuse=True), inputs=rnn_output,
                                                   att_scores=tf.transpose(scores_j_t, perm=[0, 2, 1]),
                                                   sequence_length=tf.squeeze(self.sl,
                                                                              ), dtype=tf.float32, scope="gru2")
        # hist_j_t = tf.expand_dims(hidden_state_j, axis=1)
        hist_j_t = tf.layers.batch_normalization(inputs=hidden_state_j, reuse=True)
        hist_j_t = tf.reshape(hist_j_t, [-1, hidden_units])
        hist_j_t = tf.layers.dense(hist_j_t, hidden_units, name='hist_fcn_t', reuse=True)

        u_emb_i = hist_i_t
        u_emb_j = hist_j_t
        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())
        # -- fcn begin -------
        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        # din_i = tf.concat([u_emb_i, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        # din_j = tf.concat([u_emb_j, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
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
        self.loss += aux_loss

        trainable_params = tf.trainable_variables()
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.opt = tf.train.AdamOptimizer()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run(
            [self.loss, self.train_op,], feed_dict={
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


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def target_attention(queries, keys, values, keys_length):
    """
    queries:     [B, H] current ad
    keys:        [B, T, H] history
    keys_length: [B]
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])

    outputs = tf.reduce_sum(tf.multiply(queries, keys), axis=-1)  # matmul
    outputs = tf.reshape(outputs, [-1, 1, tf.shape(keys)[1]])

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    # Activation
    scores = tf.nn.softmax(outputs)  # [B, 1, T]
    # Weighted sum
    outputs = tf.matmul(outputs, values)  # [B, 1, H]

    return scores


def self_attention(queries, keys, values, keys_length):
    """
    queries:     [B, H] current ad
    keys:        [B, T, H] history + target
    keys_length: [B]
    """
    keys_T = tf.transpose(keys, perm=[0, 2, 1])
    outputs = tf.matmul(queries, keys_T)

    # Mask_Matrix
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    diag = tf.matrix_diag(tf.cast(key_masks, dtype=tf.float32))
    key_masks = tf.tile(key_masks, [1, tf.shape(key_masks)[1]])
    key_masks = tf.reshape(key_masks, [-1, tf.shape(keys)[1], tf.shape(keys)[1]])
    key_masks = tf.cast(key_masks, tf.float32)
    key_masks = tf.matmul(diag, key_masks)
    key_masks = tf.cast(key_masks, tf.bool)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    outputs = tf.nn.softmax(outputs, axis=1)

    outputs = tf.reduce_sum(tf.matmul(outputs, values), axis=1, keep_dims=True)
    return outputs


def sum_pooling(keys, keys_length):
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    key_masks = tf.expand_dims(key_masks, 1)
    key_masks = tf.cast(key_masks, dtype=tf.float32)
    outputs = tf.matmul(key_masks, keys)
    return outputs


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
