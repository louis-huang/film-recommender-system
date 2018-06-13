#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 12:46:32 2018

@author: louis
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
#data = pd.read_csv("film_merged.csv")
#print(data.isnull().sum())
'''
## drop film with no useful feature data or not??
np.where(data.genres.isnull() & data.directors.isnull() & data.actors.isnull() & data.keywords.isnull())
'''
## read data
'''
links = pd.read_csv('links_small.csv')
#drop na data
links.dropna(axis = 0, inplace = True)
ur = pd.read_csv("ratings_small.csv")
idf = pd.read_csv("idf.csv")
'''

## test on user1
def rerank(user_id, ur, idf, user_pred, film_data, content_id, show_rating):
    clf, profile_dict = build_model(user_id, ur, idf, film_data, show_rating)
    not_rankable = []
    for index, row in user_pred.iterrows():
        cur_id = row.movieId
        if cur_id not in content_id:
            not_rankable.append(index)
    rankable = user_pred.drop(user_pred.index[not_rankable]).reset_index(drop = True)
    rankable.index = range(len(rankable))
    res = pd.merge(rankable, film_data)
    x_value = build_film_overlap(res, profile_dict)
    pred2 = clf.predict(x_value)
    good_idx = np.where(pred2 == 1)
    sim = x_value.sum(axis = 1).values
    rankable.pred_rating = sim * rankable.pred_rating
    return rankable, good_idx


def build_film_overlap(dt, profile_dict):
    film_overlap = []
    for index, row in dt.iterrows():
        over_lap = []
        for c in ['genres', 'keywords','directors', 'actors']:
            f_overlap = 0
            s = row[c]
            if pd.isnull(s): 
                f_overlap = 0
            else:
                for i in s.split("|"):
                    f_overlap += profile_dict[c][i]
            over_lap.append(f_overlap)
        film_overlap.append(over_lap)
    test = pd.DataFrame(film_overlap, columns = ['genres', 'keywords','directors', 'actors'])
    #normalize all features overlap coef
    def f(x):
        if x.max() == 0:
            return 0
        return (x - x.min()) * 4/(x.max() - x.min()) + 1
    test = test.apply(f, axis = 0)
    return test

def build_model(user_id, ur, idf, data, show_rating):
    user_id = int(user_id)
    user1 = ur.iloc[np.where(ur.userId == user_id)]
    if show_rating:
        print("No.{} user's all ratings:".format(user_id))
        print(user1)
    user_merged = pd.merge(user1, data)
    #user1.tmdbId = user1.tmdbId.astype('int64')
    #def merge_feature(dt, col):
    profile = []
    profile_dict = dict()
    for c in ['genres', 'keywords','directors', 'actors']:
        cur_dict = defaultdict(float)
        for index, row in user_merged.iterrows():
            features = row[c]
            if pd.isnull(features): continue
            for i in features.split("|"):
                cur_dict[i] += user_merged.rating[index]
        if c == 'genres':
            for k,v in cur_dict.items():
                cur_dict[k] = v * idf[k].values[0]
        profile.append(list(cur_dict.items()))
        profile_dict[c] = cur_dict
    
    test = build_film_overlap(user_merged, profile_dict)
    test.rating = user_merged.rating
    kf = StratifiedKFold(n_splits = 5, random_state = 22)
    ## if consider it as a classification problem
    #test['label'] = test.rating.apply(lambda x: 1 if x > test.rating.mean() else 0)
    #test['label'] = test.rating.apply(lambda x: 1 if x >= 3 else 0)
    #if test['label'].sum() < 5 or test['label'].sum() > (len(test) - 5):
    test['label'] = test.rating.apply(lambda x: 1 if x >= test.rating.mean() else 0)
    
    new_y = test.label.values
    new_x = test[['genres', 'keywords','directors', 'actors']].values
    ##try seven classifiers
    clfs = [svm.SVC(probability=True),DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), linear_model.LogisticRegression(),GaussianNB(), KNeighborsClassifier()  ]
    max_score = [-1,-1]
    
    for i in range(len(clfs)):
        clf = clfs[i]
        score = cross_val_score(clf, new_x, new_y, cv = kf, scoring = 'precision')
        score = score.mean()
        if score > max_score[1]:
            max_score[1] = score
            max_score[0] = i
    
    clf = clfs[max_score[0]]
    clf.fit(new_x, new_y)
    return clf, profile_dict

