#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:09:30 2018

@author: louis
"""

import pandas as pd
import numpy as np
import ast
import nltk
from collections import defaultdict
from nltk.corpus import wordnet
mf = pd.read_csv("film.csv")
keywords = pd.read_csv("keywords_2.csv")
##clean keywords
##1 root words
def root_collect(dt):
    col = 'keywords'
    PS = nltk.stem.PorterStemmer()
    keyw_root = defaultdict(set) #save root and its all derivatives
    replace_dict = dict() # key is root, value is the word to change to
    for i in dt[col]:
        if pd.isnull(i): continue
        for s in i.split("|"):
            s = s.lower()
            root = PS.stem(s)
            keyw_root[root].add(s)
    for i in keyw_root.keys():
        if len(keyw_root[i]) > 1:
            min_length = 1000
            for k in  keyw_root[i]:
                if len(k) < min_length:
                    replace = k
                    min_length = len(k)
            #replace all words that stem to i as repalce
            replace_dict[i] = replace
        else:
            replace_dict[i] = list(keyw_root[i])[0]
    return replace_dict, keyw_root
replace_keyw, key_root = root_collect(mf)

def repalce_keywords(dt, repalce_dict):
    df = dt.copy()
    PS = nltk.stem.PorterStemmer()
    for i in range(len(df)):
        keyw = df.loc[i,'keywords']
        if pd.isnull(keyw): continue
        word_list = []
        for s in keyw.split("|"):
            cur_root = PS.stem(s)
            if cur_root in repalce_dict.keys():
                word_list.append(repalce_dict[cur_root])
            else:
                word_list.append(s)
        df.set_value(i, 'keywords',"|".join(word_list))
    return df
df_replaced = repalce_keywords(mf, replace_keyw)
##2 group synonyms
def count_freq(dt, col):
    count_dict = defaultdict(int)
    for i in range(len(dt)):
        keyw = dt.loc[i,col]
        if pd.isnull(keyw):continue
        for j in keyw.split("|"):
            count_dict[j] += 1
    return count_dict
def stat(foo):
    foo_set = set(foo.keys())
    foo_list = list(foo.items())
    foo_list.sort(key = lambda x:x[1], reverse = True)
    return foo_set, foo_list

before_freq = count_freq(mf, 'keywords')
key_freq = count_freq(df_replaced, 'keywords')
all_keywords = set(key_freq.keys())
key_freq_list = list(key_freq.items())

## function to get all synonymes from wordnet
def get_syn(target):
    find = set()
    for ss in wordnet.synsets(target):
        for w in ss.lemma_names():
            index = ss.name().find('.')+1
            #just consider nouns
            if ss.name()[index] == 'n': 
                find.add(w.lower().replace('_',' '))
    return find
'''
#test
mot_cle = 'alien'
lemma = get_syn(mot_cle)
for s in lemma:
    print(' "{:<30}" in keywords list -> {} {}'.format(s, s in all_keywords,
                                                key_freq[s] if s in all_keywords else 0 ))
'''
## function to check if the lemma is in keywords and compared to a specific number
def test_keyword(target, freq, threshold):
    return freq.get(target, 0) >= threshold
key_freq_list.sort(key = lambda x:x[1], reverse = True)
repalcement = dict()
ncount = 0
## Only check those words with frequencies lower than 5.
for index, [target, freq] in enumerate(key_freq_list):
    if freq > 5: continue
    lemma = get_syn(target)
    if len(lemma) == 0: continue
    #find all synonymes that have more frequences than itself
    syn_list = [(s, key_freq[s]) for s in lemma if test_keyword(s, key_freq, key_freq[target])]
    syn_list.sort(key = lambda x:(x[1], x[0]), reverse = True)
    if len(syn_list) <= 1: continue
    if target == syn_list[0][0]: continue
    ncount+=1
    if ncount < 8:
        print('{:<12} -> {:<12} (init: {})'.format(target, syn_list[0][0], syn_list))    
    repalcement[target] = syn_list[0][0]
## be careful some keys will also in values
ncount = 0
for k, v in repalcement.items():
    if v in repalcement.keys():
        repalcement[k] = repalcement[v]
        ncount+=1
#replace
df_new = repalce_keywords(df_replaced, repalcement)
new_key_freq = count_freq(df_new, 'keywords')
new_key_freq_list = list(new_key_freq.items())
#delete low freq keywords
def delete_low(dt, freq_dict):
    df = dt.copy()
    for index, row in df.iterrows():
        words = row['keywords']
        if pd.isnull(words):continue
        word_list = []
        for s in words.split("|"):
            if freq_dict.get(s, 4) > 3: word_list.append(s)
        df.set_value(index, "keywords", "|".join(word_list))
    return df

df_new2 = delete_low(df_new, new_key_freq)

final_key_freq = count_freq(df_new2, 'keywords')
final_key_set, final_key_list = stat(final_key_freq)
print(final_key_list[:4])
print(len(final_key_set))

##find duplicates and delete duplicates
id_idx = defaultdict(list)
for idx, row in df_new2.iterrows():
    id_idx[row.id].append(idx)
idx_keep = []
for k, v in id_idx.items():
    idx_keep.append(v[0])
##remove duplicates
data_dup = df_new2.iloc[idx_keep]
##save new dataframe


data_dup.to_csv("film_cleaned.csv", index = None)
#delete actors whose freq is lower than 2??? tbd
## for our problem we only have 9000 films so we need to filter first
#merge with our films
links = pd.read_csv("links_small.csv")
meta = pd.read_csv("movies_metadata.csv")
tbim = meta[['id','imdb_id']]
wrong_format = []
for index, row in tbim.iterrows():
    try:
        int(row.id)
    except ValueError:
        wrong_format.append(index)
new_tbim = tbim.drop(tbim.index[wrong_format])
new_tbim.id = new_tbim.id.astype('int64')
merge1 = pd.merge(links, new_tbim, left_on = 'tmdbId', right_on = 'id', how = 'left')
merge2 = pd.merge(links, data_dup, left_on = 'tmdbId', right_on = 'id', how = 'left')
merge_droped = merge2.drop(np.where(merge2.title.isnull())[0])
merge_droped.drop('id', axis = 1, inplace = True)            
merge_droped.to_csv('film_merged.csv', index = None)
    
