import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf

from input_tin import DataInput, DataInputTest
from tin import Model
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100
all_train = []
all_test = []
all_eval = []

with open(
        'dataset1.pkl', 'rb') as f:
    train_set = pickle.load(f)
    for i in range(len(train_set)):
        tmp = copy.deepcopy(train_set[i])
        l = len(tmp[1])
        # here is an example
        # (115053, [2472, 12254, 40489, 40491, 42250, 42264, 7681], 29205, 1, [7, 6, 5, 4, 3, 2, 1], 0)
        # (用户id，[历史广告id序列]，目标广告id，是否被点击，[历史广告位置序列]，目标广告位置)
        a = (tmp[0], tmp[1], tmp[2], tmp[3], [l - x for x in range(l)], 0)
        all_train.append(a)
    train_set = all_train  # len=2223958

    test_set = pickle.load(f)
    for i in range(len(test_set)):
        tmp = copy.deepcopy(test_set[i])
        l = len(tmp[1])
        # here is an example
        # (180873, [4763, 29004, 24347, 20887], (37639, 39715), [4, 3, 2, 1], 0)
        a = (tmp[0], tmp[1], tmp[2], [l - x for x in range(l)], 0)
        all_test.append(a)
    test_set = all_test  # len=192403

    eval_set = pickle.load(f)
    for i in range(len(eval_set)):
        tmp = copy.deepcopy(eval_set[i])
        l = len(tmp[1])
        # here is an example
        # (34945, [31709, 37167, 15260, 60902, 50938], (62221, 2141), [5, 4, 3, 2, 1], 0)
        a = (tmp[0], tmp[1], tmp[2], [l - x for x in range(l)], 0)
        all_eval.append(a)
    eval_set = all_eval  # len=192403

    cate_list = pickle.load(f)  # 63001--each item's category
    user_count, item_count, cate_count = pickle.load(f)  # 192403 63001 801

best_auc = 0.0
test_gauc = 0.0
test_logloss = 0.0


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


def _eval(sess, model):
    eval_auc_sum = 0.0
    for _, uij in DataInputTest(eval_set, test_batch_size):
        auc_, cross_ = model.eval(sess, uij)
        eval_auc_sum += auc_ * len(uij[0])
    eval_gauc = eval_auc_sum / len(eval_set)
    global best_auc
    global test_gauc
    global test_logloss
    if best_auc < eval_gauc:
        test_auc_sum = 0.0
        test_loss_sum = 0.0
        for _, uij in DataInputTest(test_set, test_batch_size):
            auc_, cross_ = model.eval(sess, uij)
            test_loss_sum += cross_
            test_auc_sum += auc_ * len(uij[0])
        test_gauc = test_auc_sum / len(test_set)
        test_logloss = test_loss_sum / (2 * len(test_set))
        best_auc = eval_gauc
        model.save(sess, 'save/model_tin.ckpt')
    return eval_gauc


gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(item_count, cate_count, cate_list)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    sys.stdout.flush()
    lr = 1.0
    start_time = time.time()
    for _ in range(4):
        random.shuffle(train_set)  # 最后的train_set不是同一个
        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, train_batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss

            if model.global_step.eval() % 1000 == 0:
                eval_gauc = _eval(sess, model)
                print(
                    'Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tTest_Logloss: %.4f\tTest_GAUC: %.4f' %
                    (model.global_epoch_step.eval(), model.global_step.eval(), loss_sum / 1000, eval_gauc, test_logloss,
                     test_gauc))
                sys.stdout.flush()
                loss_sum = 0.0

            if model.global_step.eval() % 336000 == 0:
                lr = 0.1

        print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time() - start_time))
        print('Current Model on the test set GAUC:%.4f\tLogloss:%.4f' % (test_gauc, test_logloss))
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
    print('Model on the test set GAUC:%.4f\tLogloss:%.4f' % (test_gauc, test_logloss))
    sys.stdout.flush()
