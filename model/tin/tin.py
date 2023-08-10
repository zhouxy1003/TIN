import tensorflow as tf


class Model(object):

    def __init__(self, item_count, cate_count, cate_list):
        self.u = tf.placeholder(tf.int32, [None, ])  # [B] user
        self.i = tf.placeholder(tf.int32, [None, ])  # [B] pos
        self.j = tf.placeholder(tf.int32, [None, ])  # [B] neg
        self.y = tf.placeholder(tf.float32, [None, ])  # [B] label
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]  history item
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B] sequence length
        self.lr = tf.placeholder(tf.float64, [])  # learning rate
        self.pos = tf.placeholder(tf.int32, [None, ])  # position of target item
        self.pos_i = tf.placeholder(tf.int32, [None, None])  # position of history sequence

        hidden_units = 128  # values

        position = tf.get_variable("pos", shape=[431, hidden_units])  # time embedding 431 buckets
        item = tf.get_variable("item", [item_count, hidden_units // 2])
        cate = tf.get_variable("cate", [cate_count, hidden_units // 2])  # item_emb_w
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        ic = tf.gather(cate_list, self.i)
        i_b = tf.gather(item_b, self.i)
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.i),
            tf.nn.embedding_lookup(cate, ic),
        ], axis=1)
        i_pos_emb = tf.nn.embedding_lookup(position, self.pos)
        i_emb = tf.add(i_emb, i_pos_emb)  # [bs, d]

        jc = tf.gather(cate_list, self.j)
        j_b = tf.gather(item_b, self.j)
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item, self.j),
            tf.nn.embedding_lookup(cate, jc),
        ], axis=1)
        j_pos_emb = i_pos_emb
        j_emb = tf.add(j_emb, j_pos_emb)  # [bs, d]

        i_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.i),  # item_target_q
            tf.nn.embedding_lookup(cate, ic),  # cate_target_q
        ], axis=1)
        i_pos_tar = tf.nn.embedding_lookup(position, self.pos)  # position_target
        i_t_q = tf.add(i_t_q, i_pos_tar)  # why same as i_emb......

        j_t_q = tf.concat(values=[
            tf.nn.embedding_lookup(item, self.j),  # item_target_q
            tf.nn.embedding_lookup(cate, jc),  # cate_target_q
        ], axis=1)  # why same as j_emb......
        j_pos_tar = i_pos_tar
        j_t_q = tf.add(j_t_q, j_pos_tar)

        hc = tf.gather(cate_list, self.hist_i)
        h_t_v = tf.concat([
            tf.nn.embedding_lookup(item, self.hist_i),
            tf.nn.embedding_lookup(cate, hc),
        ], axis=2)
        h_pos = tf.nn.embedding_lookup(position, self.pos_i)
        h_t_v = tf.add(h_t_v, h_pos)

        h_t_k = tf.concat([
            tf.nn.embedding_lookup(item, self.hist_i),  # item_target_q
            tf.nn.embedding_lookup(cate, hc),  # cate_target_q
        ], axis=2)  # same as h_t_v
        h_pos_tar = tf.nn.embedding_lookup(position, self.pos_i)  # position_target
        h_t_k = tf.add(h_t_k, h_pos_tar)

        hist_i_t = target_attention(i_t_q, h_t_k, h_t_v, self.sl)  # [B, 1, H]
        hist_i_t = tf.layers.batch_normalization(inputs=hist_i_t)
        hist_i_t = tf.reshape(hist_i_t, [-1, hidden_units])  # [B, H]
        hist_i_t = tf.layers.dense(hist_i_t, hidden_units, name='hist_fcn_t')  # [B, H]

        hist_j_t = target_attention(j_t_q, h_t_k, h_t_v, self.sl)
        hist_j_t = tf.layers.batch_normalization(inputs=hist_j_t, reuse=True)
        hist_j_t = tf.reshape(hist_j_t, [-1, hidden_units])
        hist_j_t = tf.layers.dense(hist_j_t, hidden_units, name='hist_fcn_t', reuse=True)  # [B, H]

        u_emb_i = hist_i_t  # [B, H]
        u_emb_j = hist_j_t  # [B, H]

        # -- fcn begin -------
        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)  # [B, 3*H]
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')  # [B, 80]
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')  # [B, 40]
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')  # [B, 1]

        din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)

        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])  # [B]
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])  # [B]
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]，正样本预测值减负样本预测值
        self.logits = i_b + d_layer_3_i  # [B]
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)  # [B]
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)  # [B]
        self.score_i = tf.reshape(self.score_i, [-1, 1])  # [B, 1]
        self.score_j = tf.reshape(self.score_j, [-1, 1])  # [B, 1]
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)  # [B, 2]
        print(self.p_and_n.get_shape().as_list())  # [B, 2]

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
            [self.loss, self.train_op], feed_dict={
                self.u: uij[0],  # 用户id
                self.i: uij[1],  # 目标广告id序列
                self.y: uij[2],  # 是否点击目标广告
                self.hist_i: uij[3],  # 历史广告id序列
                self.sl: uij[4],  # 历史行为真实长度
                self.pos_i: uij[5],  # 历史序列位置信息（和长度对应）
                self.pos: uij[6],
                self.lr: l
            })
        return loss

    def eval(self, sess, uij):
        u_auc, cross = sess.run([self.mf_auc, self.cross], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
            self.pos_i: uij[5],
            self.pos: uij[6]
        })
        return u_auc, cross

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def target_attention(queries, keys, values, keys_length):
    """
    queries:     [B, H] current ad
    keys:        [B, T, H] history
    values:      [B, T, H] history
    keys_length: [B]
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [B, T * H]
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # [B, T, H]

    outputs = tf.reduce_sum(tf.multiply(queries, keys), axis=-1)  # 对应元素相乘 [B, T]
    outputs = tf.reshape(outputs, [-1, 1, tf.shape(keys)[1]])  # [B, 1, T]

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
