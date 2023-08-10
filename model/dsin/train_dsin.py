import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input_dsin import DataInput, DataInputTest
from model_dsin import Model
import copy

os.system('pip install tqdm')

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1234)  # 2021
np.random.seed(1234)  # 2021
tf.set_random_seed(1234)  # 2021

train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100
all_train = []
all_test = []
all_eval = []

with open(
        '/cephfs/group/file-teg-datamining-wx-dm-intern/lukahlzhou/ft_local/DeepInterestNetwork-master/dsin/ft_local/dataset1_dsin.pkl',
        'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    eval_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return 1.0 - auc / (2.0 * tp2 * fp2)
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    # print "============== p ============="
    # print score_p
    # print "============== n ============="
    # print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr

def _logits_arr(logits):
    logits_p = logits[:, 0]
    logits_n = logits[:, 1]
    p = tf.ones_like(logits_p)
    n = tf.zeros_like(logits_n)
    cross_p = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_p, labels=p)
    cross_n = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_n, labels=n)
    cross = tf.reduce_sum(cross_p)+tf.reduce_sum(cross_n)
    # print(cross.eval())
    return cross.eval()


def _eval(sess, model):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(eval_set, test_batch_size):
        auc_, score_, cross_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    eval_gauc = auc_sum / len(eval_set)
    Auc = calc_auc(score_arr)
    global best_auc
    global final_test_gauc
    global final_test_logloss
    if best_auc < eval_gauc:
        test_auc_sum = 0.0
        cross_sum = 0.0
        test_score_arr = []
        for _, uij in DataInputTest(test_set, test_batch_size):
            auc_, score_, cross_ = model.eval(sess, uij)
            test_score_arr += _auc_arr(score_)
            cross_sum += cross_
            test_auc_sum += auc_ * len(uij[0])
        test_gauc = test_auc_sum / len(test_set)
        cross = cross_sum / (2*len(test_set))
        test_Auc = calc_auc(test_score_arr)
        final_test_gauc = test_gauc
        final_test_logloss = cross
        best_auc = eval_gauc
    return eval_gauc, Auc


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(item_count, cate_count, cate_list)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('eval_gauc: %.4f\t eval_auc: %.4f' % _eval(sess, model))
    sys.stdout.flush()
    lr = 1.0
    start_time = time.time()
    for _ in range(20):

        random.shuffle(train_set)  # 最后的train_set不是同一个

        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss

            if model.global_step.eval() % 1000 == 0:
                eval_gauc, Auc = _eval(sess, model)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f\tBest_Logloss: %.4f' %
                      (model.global_epoch_step.eval(), model.global_step.eval(),
                       loss_sum / 1000, eval_gauc, Auc, final_test_logloss))
                sys.stdout.flush()
                loss_sum = 0.0

            if model.global_step.eval() % 336000 == 0:
                lr = 0.1

        print('Epoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time() - start_time))
        print('Current Model on the test set GAUC:%.4f\tLogloss:%.4f'% (final_test_gauc, final_test_logloss))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
    print('Model on the test set GAUC:%.4f\tLogloss:%.4f'% (final_test_gauc, final_test_logloss))
    sys.stdout.flush()
    print('best eval_gauc:', best_auc)
    sys.stdout.flush()