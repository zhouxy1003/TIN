import random
import pickle
import numpy as np
import random
import pandas as pd
import copy

random.seed(1234)
from tqdm import tqdm

with open("../raw_data/remap.pkl", "rb") as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
eval_set = []
test_set = []
for reviewerID, hist in tqdm(reviews_df.groupby("reviewerID")):
    pos_list = hist["asin"].tolist()

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist = pos_list[:i]
        if i < len(pos_list) - 2:
            train_set.append((reviewerID, hist, pos_list[i], 1))
            train_set.append((reviewerID, hist, neg_list[i], 0))
        elif i == len(pos_list) - 2:
            label = (pos_list[i], neg_list[i])
            eval_set.append((reviewerID, hist, label))
        elif i == len(pos_list) - 1:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist, label))

random.shuffle(train_set)
random.shuffle(eval_set)
random.shuffle(test_set)

assert len(test_set) == user_count

with open("dataset.pkl", "wb") as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(eval_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
