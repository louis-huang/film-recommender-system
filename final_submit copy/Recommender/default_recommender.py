#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 23:48:13 2018

@author: louis
"""


from surprise import Dataset, Reader
import hybrid

from collections import defaultdict
from sigweight import KNNSigWeighting
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

def predict(path, user_id = -1, predict_all = False):
    ##read data and transform it to
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file("{}".format(path), reader = reader)
    user_knn_weight = KNNSigWeighting(k = 3,sim_options={'user_based': True})
    item_knn_weight = KNNSigWeighting(k = 3, sim_options={'user_based': False})
    hybrid_knn = hybrid.WeightedHybrid([item_knn_weight, user_knn_weight])
    all_train = data.build_full_trainset()
    hybrid_knn.fit(all_train)
    all_test = all_train.build_anti_testset()
    if predict_all == True:
        predictions = hybrid_knn.test(all_test)
        pred = get_top_n(predictions, 100)
        #save all predictions
        with open("new_predictions.pickle","wb") as f:
            pickle.dump(pred, f, protocol=2)
        f.close()
        # save my predictions specifically
        predictions = hybrid_knn.testme(str(user_id), all_test)
        pred = get_top_n(predictions, 100)
        with open("my_predictions.pickle","wb") as f:
            pickle.dump(pred, f, protocol=2)
        f.close()
        print("Done updating all users' recommendations!!!")
        return
    if user_id != -1:
        predictions = hybrid_knn.testme(str(user_id), all_test)
        pred = get_top_n(predictions, 100)
        with open("my_predictions.pickle","wb") as f:
            pickle.dump(pred, f, protocol=2)
        f.close()
        print("Done recommending you!!!")
    else:
        predictions = hybrid_knn.test(all_test)
        pred = get_top_n(predictions, 100)
        with open("old_predictions.pickle","wb") as f:
            pickle.dump(pred, f, protocol=2)
        f.close()
        print("Done recommending on original data")
    #save my result
    





