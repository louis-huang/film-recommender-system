#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 23:19:39 2018

@author: louis
"""
from surprise import SVD, BaselineOnly
from collections import defaultdict
from surprise import Dataset, Reader
import pickle

def get_top_n(predictions, n=10):
   
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def predict(path):
    ##read data and transform it to
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file("{}".format(path), reader = reader)
    all_train = data.build_full_trainset()
    bsl = BaselineOnly()
    svd = SVD()
    bsl.fit(all_train)
    svd.fit(all_train)
    all_test = all_train.build_anti_testset()
    bsl_predictions = bsl.test(all_test)
    bsl_pred = get_top_n(bsl_predictions, 100)
    svd_predictions = bsl.test(all_test)
    svd_pred = get_top_n(svd_predictions, 100) 
    with open("baseline_predictions.pickle","wb") as f:
        pickle.dump([bsl_pred, svd_pred], f, protocol=2)
    f.close()
    print("Done recommending using baseline model and SVD model.")
    