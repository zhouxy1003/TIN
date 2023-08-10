import tensorflow as tf

# from Dice import dice
import tensorflow.contrib as contrib


# from .sequence import BiLSTM


class Model(object):

    def __init__(self, item_count, cate_count, cate_list):
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sess_i = tf.placeholder(tf.int32,
                                     [None, None, None])  # [B, M, T], M session number fixed as 10, K session length
        self.pos_k = tf.placeholder(tf.int32, [None, None, None])  # k-th session [1-10]
        self.pos_t = tf.placeholder(tf.int32, [None, None, None])  # t-th behavior in the session
        self.sess_sl = tf.placeholder(tf.int32, [None, None, ])  # [B, M] padding in session
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B] padding in sequence
        self.lr = tf.placeholder(tf.float64, [])
        self.pos_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        hidden_units = 128  # values
        # hidden_units_copy = 128  # keys, queries

        position_k = tf.get_variable("pos_k", shape=[10, hidden_units])
        position_t = tf.get_variable("pos_t", shape=[431, hidden_units])

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
            tf.nn.embedding_lookup(item, self.i),  # item_target_q
            tf.nn.embedding_lookup(cate, ic),  # cate_target_q
        ], axis=1)

        j_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.j),  # item_target_q
            tf.nn.embedding_lookup(cate, jc),  # cate_target_q
        ], axis=1)

        hc = tf.gather(cate_list, self.sess_i)

        s_t = tf.concat([
            tf.nn.embedding_lookup(item, self.sess_i),
            tf.nn.embedding_lookup(cate, hc),
        ], axis=3)  # 2→3
        s_pos_k = tf.nn.embedding_lookup(position_k, self.pos_k)
        s_pos_t = tf.nn.embedding_lookup(position_t, self.pos_t)
        s_t = s_t + s_pos_k + s_pos_t

        # DSIN begins
        # sess_fea = dsin_target_attention(i_t_q, s_t, s_t, self.sess_sl, hidden_units)  # dsin_self_attention
        sess_fea = dsin_self_attention(s_t, s_t, s_t, self.sess_sl, hidden_units)  # dsin_self_attention
        lstm_outputs = BiLSTM(hidden_units, sess_fea, tf.squeeze(self.sl))

        interest_attention_layer_i = target_attention(i_t_q, sess_fea, sess_fea, self.sl)
        lstm_attention_layer_i = target_attention(i_t_q, lstm_outputs, lstm_outputs, self.sl)
        user_interest_i = tf.concat([interest_attention_layer_i, lstm_attention_layer_i], axis=-1)
        # hist_i_t = tf.expand_dims(user_interest_i, axis=1)
        hist_i_t = tf.layers.batch_normalization(inputs=user_interest_i)
        hist_i_t = tf.reshape(hist_i_t, [-1, 2 * hidden_units])  # hidden_units
        hist_i_t = tf.layers.dense(hist_i_t, hidden_units, name='hist_fcn_t')
        interest_attention_layer_j = target_attention(j_t_q, sess_fea, sess_fea, self.sl)
        lstm_attention_layer_j = target_attention(j_t_q, lstm_outputs, lstm_outputs, self.sl)
        user_interest_j = tf.concat([interest_attention_layer_j, lstm_attention_layer_j], axis=-1)

        # hist_j_t = tf.expand_dims(user_interest_j, axis=1)
        hist_j_t = tf.layers.batch_normalization(inputs=user_interest_j, reuse=True)
        hist_j_t = tf.reshape(hist_j_t, [-1, 2 * hidden_units])  # hidden_units
        hist_j_t = tf.layers.dense(hist_j_t, hidden_units, name='hist_fcn_t', reuse=True)

        # u_emb_i = tf.reshape(user_interest_i, [-1, 2*hidden_units]) # hist_i_t
        # u_emb_j = tf.reshape(user_interest_j, [-1, 2*hidden_units]) # hist_j_t
        u_emb_i = hist_i_t
        u_emb_j = hist_j_t
        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())
        # -- fcn begin -------
        dsin_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        # dsin_i = tf.concat([u_emb_i, i_emb], axis=-1)
        dsin_i = tf.layers.batch_normalization(inputs=dsin_i, name='b1')
        d_layer_1_i = tf.layers.dense(dsin_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        dsin_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        # dsin_j = tf.concat([u_emb_j, j_emb], axis=-1)
        dsin_j = tf.layers.batch_normalization(inputs=dsin_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(dsin_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
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
                self.sess_i: uij[3],
                self.sl: uij[4],
                self.pos_k: uij[5],
                self.pos_t: uij[6],
                self.sess_sl: uij[7],
                self.lr: l,
            })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n, cross = sess.run([self.mf_auc, self.p_and_n, self.cross], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.sess_i: uij[3],
            self.sl: uij[4],
            self.pos_k: uij[5],
            self.pos_t: uij[6],
            self.sess_sl: uij[7],
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
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]
    # Weighted sum
    outputs = tf.matmul(outputs, values)  # [B, 1, H]

    return outputs


def dsin_target_attention(queries, keys, values, sess_sl, d):
    """
    queries:     [B, H] → [B*M, T, H]
    keys:        [B, M, T, H] → [B*M, T, H]
    keys_length: [B, M, ] → [B*M, ]
    """
    sess_num = tf.shape(keys)[1]
    queries = tf.tile(queries, [1, sess_num])  # [B, M*H]
    queries = tf.reshape(queries, [-1, sess_num, d])  # [B, M, H]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[2]])  # [B, M, T*H]
    queries = tf.reshape(queries, [-1, sess_num, tf.shape(keys)[2], d])  # [B, M, T, H]

    queries = tf.reshape(queries, [-1, tf.shape(queries)[2], d])
    keys = tf.reshape(keys, [-1, tf.shape(keys)[2], d])
    values = tf.reshape(values, [-1, tf.shape(values)[2], d])
    keys_length = tf.reshape(sess_sl, [-1])

    outputs = tf.reduce_sum(tf.multiply(queries, keys), axis=-1)  # matmul [B*M, T]
    outputs = tf.reshape(outputs, [-1, 1, tf.shape(keys)[1]])  # matmul [B*M, 1, T]

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B*M, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B*M, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B*M, 1, T]

    # Scale
    outputs = outputs / (d ** 0.5)
    # Activation
    outputs = tf.nn.softmax(outputs)  # [B*M, 1, T]
    # Weighted sum
    outputs = tf.matmul(outputs, values)  # [B*M, 1, T] * [B*M, T, H] = [B*M, 1, H]
    outputs = tf.reshape(outputs, [-1, sess_num, d])

    return outputs


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
    outputs = tf.matmul(outputs, values)
    # outputs = tf.reduce_sum(tf.matmul(outputs, values), axis=1, keep_dims=True)
    return outputs


def dsin_self_attention(queries, keys, values, sess_sl, d):
    """
    queries:     [B, M, T, H] → [B*M, T, H]
    keys:        [B, M, T, H]
    keys_length: [B, M, ] → [B*M, ]
    """
    sess_num = tf.shape(keys)[1]
    # d = keys.get_shape().as_list()[-1]
    queries = tf.reshape(queries, [-1, tf.shape(queries)[2], d])
    keys = tf.reshape(keys, [-1, tf.shape(keys)[2], d])
    values = tf.reshape(values, [-1, tf.shape(values)[2], d])
    # print(keys_length)
    # keys_length = tf.reshape(keys_length, [tf.shape(keys_length)[0]*tf.shape(keys_length)[1]])
    keys_length = tf.reshape(sess_sl, [-1])
    # print(keys_length)
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
    outputs = outputs / (d ** 0.5)
    outputs = tf.nn.softmax(outputs, axis=1)
    paddings = tf.zeros_like(outputs)
    outputs = tf.where(key_masks, outputs, paddings)
    outputs = tf.matmul(outputs, values)

    outputs = tf.reshape(outputs, [-1, sess_num, tf.shape(queries)[1], d])
    outputs = tf.reduce_sum(outputs, axis=2)  # [B, M, H], session_number, divide sess_sl
    sess_paddings = tf.ones_like(sess_sl)
    sess_masks = tf.cast(sess_sl, tf.bool)
    sess_sl = tf.where(sess_masks, sess_sl, sess_paddings)
    # sess_sl[sess_sl==0] = 1
    outputs = tf.div(outputs, tf.cast(tf.tile(tf.expand_dims(sess_sl, 2), [1, 1, d]), tf.float32))
    # print(outputs.get_shape())
    # outputs = tf.reduce_sum(tf.matmul(outputs, values), axis=1, keep_dims=True)
    return outputs


def sum_pooling(keys, keys_length):
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    key_masks = tf.expand_dims(key_masks, 1)
    key_masks = tf.cast(key_masks, dtype=tf.float32)
    outputs = tf.matmul(key_masks, keys)
    return outputs


def BiLSTM(hidden_units, input, seqlen):
    lstm_cell_fw = contrib.rnn.LSTMCell(hidden_units, name='fw', reuse=tf.AUTO_REUSE)
    lstm_cell_bw = contrib.rnn.LSTMCell(hidden_units, name='bw', reuse=tf.AUTO_REUSE)
    out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw, inputs=input,
                                                 sequence_length=seqlen, dtype=tf.float32)
    out = (out[0] + out[1]) / 2
    return out
