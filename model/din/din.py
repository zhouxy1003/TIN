import tensorflow as tf


class Model(object):

    def __init__(self, item_count, cate_count, cate_list):
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128

        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        ic = tf.gather(cate_list, self.i)
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
        i_b = tf.gather(item_b, self.i)

        jc = tf.gather(cate_list, self.j)
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        j_b = tf.gather(item_b, self.j)

        hc = tf.gather(cate_list, self.hist_i)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

        hist_i = attention(i_emb, h_emb, self.sl)
        # -- attention end ---

        hist_i = tf.layers.batch_normalization(inputs=hist_i)
        hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')

        u_emb_i = hist_i

        hist_j = attention(j_emb, h_emb, self.sl)
        # -- attention end ---

        hist_j = tf.layers.batch_normalization(inputs=hist_j, reuse=True)
        hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
        hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)

        u_emb_j = hist_j
        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())

        # -- fcn begin -------
        # din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)  # different from DIN+
        din_i = tf.concat([u_emb_i, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

        # din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)  # different from DIN+
        din_j = tf.concat([u_emb_j, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        self.logits = i_b + d_layer_3_i

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
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.y: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.lr: l,
        })
        return loss

    def eval(self, sess, uij):
        u_auc, cross = sess.run([self.mf_auc, self.cross], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, cross

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def attention(queries, keys, keys_length):
    '''
      queries:     [B, H] current ad
      keys:        [B, T, H] history
      keys_length: [B]
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])

    # Ablation study of DIN
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # din_all = tf.concat([queries], axis=-1)
    # din_all = tf.concat([keys], axis=-1)
    # din_all = tf.concat([queries - keys], axis=-1)
    # din_all = tf.concat([queries * keys], axis=-1)

    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all

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
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs
