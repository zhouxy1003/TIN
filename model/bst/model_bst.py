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
        self.pos = tf.placeholder(tf.int32, [None, ])
        self.pos_i = tf.placeholder(tf.int32, [None, None])
        self.pos_new = tf.placeholder(tf.int32, [None, None])  # pos_i + 0
        self.i_new = tf.placeholder(tf.int32, [None, None])  # hist_i + i
        self.j_new = tf.placeholder(tf.int32, [None, None])  # hist_i + j
        self.sl_new = tf.placeholder(tf.int32, [None, ])  # [B]
        hidden_units = 128  # values

        position_emb_w = tf.get_variable("pos_emb", shape=[431, hidden_units])

        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])

        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))

        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])  # hidden_units // 2

        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        ic = tf.gather(cate_list, self.i)

        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
        i_pos_emb = tf.nn.embedding_lookup(position_emb_w, self.pos)
        # i_emb = tf.add(i_emb, i_pos_emb)

        i_b = tf.gather(item_b, self.i)  # initialized by a constant, bias, wx+b

        jc = tf.gather(cate_list, self.j)

        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        j_pos_emb = i_pos_emb
        # j_emb = tf.add(j_emb, j_pos_emb)

        j_b = tf.gather(item_b, self.j)

        # -- self begins ---
        hc_i = tf.gather(cate_list, self.i_new)
        hc_j = tf.gather(cate_list, self.j_new)

        h_emb_i = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i_new),
            tf.nn.embedding_lookup(cate_emb_w, hc_i),
        ], axis=2)

        h_pos_self = tf.nn.embedding_lookup(position_emb_w, self.pos_new)
        h_emb_i = tf.add(h_emb_i, h_pos_self)

        h_emb_j = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j_new),
            tf.nn.embedding_lookup(cate_emb_w, hc_j),
        ], axis=2)
        h_emb_j = tf.add(h_emb_j, h_pos_self)
        # -- self ends ---

        hist_i = self_attention(h_emb_i, self.sl_new, 1, hidden_units)
        u_emb_i = hist_i

        hist_j = self_attention(h_emb_j, self.sl_new, 1, hidden_units)
        u_emb_j = hist_j

        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())
        # -- fcn begin -------
        bst_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        # bst_i = tf.concat([u_emb_i, i_emb], axis=-1)
        '''
        d_layer_0_i = tf.layers.dense(din_i, 1024, activation=tf.nn.leaky_relu, name='f1')
        d_layer_1_i = tf.layers.dense(d_layer_0_i, 512, activation=tf.nn.leaky_relu, name='f2')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 256, activation=tf.nn.leaky_relu, name='f3')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f4')
        '''
        bst_i = tf.layers.batch_normalization(inputs=bst_i, name='b1')
        d_layer_1_i = tf.layers.dense(bst_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        bst_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        # bst_j = tf.concat([u_emb_j, j_emb], axis=-1)
        '''
        d_layer_0_j = tf.layers.dense(din_j, 1024, activation=tf.nn.leaky_relu, name='f1', reuse=True)
        d_layer_1_j = tf.layers.dense(d_layer_0_j, 512, activation=tf.nn.leaky_relu, name='f2', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 256, activation=tf.nn.leaky_relu, name='f3', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f4', reuse=True)
        '''
        bst_j = tf.layers.batch_normalization(inputs=bst_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(bst_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
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
            [self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.y: uij[2],
                self.hist_i: uij[3],
                self.sl: uij[4],
                self.pos_i: uij[5],
                self.pos: uij[6],
                self.pos_new: uij[7],
                self.i_new: uij[8],
                self.sl_new: uij[9],
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
            self.pos_i: uij[5],
            self.pos: uij[6],
            self.pos_new: uij[7],
            self.i_new: uij[8],
            self.j_new: uij[9],
            self.sl_new: uij[10],
        })
        return u_auc, socre_p_and_n, cross

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def vanilla_self_attention(E, keys_length, h, emb):
    """
    queries:     [B, H] current ad
    keys:        [B, T, H] history + target
    keys_length: [B]
    """
    keys = tf.layers.dense(E, emb, activation=tf.nn.relu, name='keys', reuse=tf.AUTO_REUSE)
    queries = tf.layers.dense(E, emb, activation=tf.nn.relu, name='queries', reuse=tf.AUTO_REUSE)
    values = tf.layers.dense(E, emb, activation=tf.nn.relu, name='values', reuse=tf.AUTO_REUSE)

    Q_ = tf.concat(tf.split(queries, h, axis=-1), axis=0)
    K_ = tf.concat(tf.split(keys, h, axis=-1), axis=0)
    V_ = tf.concat(tf.split(values, h, axis=-1), axis=0)

    outputs = tf.matmul(Q_, tf.transpose(K_, perm=[0, 2, 1]))

    # Mask_Matrix
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    diag = tf.matrix_diag(tf.cast(key_masks, dtype=tf.float32))
    key_masks = tf.tile(key_masks, [1, tf.shape(key_masks)[1]])
    key_masks = tf.reshape(key_masks, [-1, tf.shape(keys)[1], tf.shape(keys)[1]])
    key_masks = tf.cast(key_masks, tf.float32)
    key_masks = tf.matmul(diag, key_masks)
    key_masks = tf.cast(key_masks, tf.bool)
    key_masks = tf.tile(key_masks, [h, 1, 1])
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    outputs = tf.nn.softmax(outputs, axis=1)
    paddings = tf.zeros_like(outputs)
    outputs = tf.where(key_masks, outputs, paddings)

    outputs = tf.matmul(outputs, V_)  # MH(E)

    outputs = tf.concat(tf.split(outputs, h, axis=0), axis=2)
    outputs += E

    # normalization 层
    S = tf.contrib.layers.layer_norm(outputs)

    # BST Eqn.(6)
    S1 = tf.layers.dense(S, emb, activation=tf.nn.leaky_relu, name='W1', reuse=tf.AUTO_REUSE)
    S2 = tf.layers.dense(S1, emb, activation=None, name='W2', reuse=tf.AUTO_REUSE)
    F = tf.contrib.layers.layer_norm(S2 + S)

    # outputs = tf.reduce_sum(F, axis=1, keep_dims=True)
    outputs = tf.reduce_sum(F, axis=1)
    outputs = tf.div(outputs, tf.cast(tf.tile(tf.expand_dims(keys_length, 1), [1, emb]), tf.float32))
    return outputs


def self_attention(E, keys_length, h, emb):
    keys_T = tf.transpose(E, perm=[0, 2, 1])
    outputs = tf.matmul(E, keys_T)

    # Mask_Matrix
    key_masks = tf.sequence_mask(keys_length, tf.shape(E)[1])
    diag = tf.matrix_diag(tf.cast(key_masks, dtype=tf.float32))
    key_masks = tf.tile(key_masks, [1, tf.shape(key_masks)[1]])
    key_masks = tf.reshape(key_masks, [-1, tf.shape(E)[1], tf.shape(E)[1]])
    key_masks = tf.cast(key_masks, tf.float32)
    key_masks = tf.matmul(diag, key_masks)
    key_masks = tf.cast(key_masks, tf.bool)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)

    # Scale
    outputs = outputs / (E.get_shape().as_list()[-1] ** 0.5)
    outputs = tf.nn.softmax(outputs, axis=1)
    paddings = tf.zeros_like(outputs)
    outputs = tf.where(key_masks, outputs, paddings)
    outputs = tf.matmul(outputs, E)
    outputs = tf.reduce_sum(outputs, axis=1)
    outputs = tf.div(outputs, tf.cast(tf.tile(tf.expand_dims(keys_length, 1), [1, emb]), tf.float32))
    return outputs


def sum_pooling(keys, keys_length):
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    key_masks = tf.expand_dims(key_masks, 1)
    key_masks = tf.cast(key_masks, dtype=tf.float32)
    outputs = tf.matmul(key_masks, keys)
    return outputs
