#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 23:25:07 2018

@author: louis
"""

import pandas as pd
import numpy as np
import pickle
import rerank_1
import random
import default_recommender
import bsl_recommender

class Recommender(object):
    
    def __init__(self):
        print("Welcome to Louis's Film Recommender!")
        self.set_user_rating()
        self.read_data()
        self.set_me()
        #initiate old predictions
        self.set_recommendations()
        print("Done initializing recommender!")
        print("Type test.examples() for help if you are new.")
        
    def reset(self):
        print("Reseting......")
        #Because we reset, we just read the original user rating data.
        self.user_rating = pd.read_csv("ratings_small.csv")
        self.user_all_id = list(set(self.user_rating.userId))
        self.read_data()
        self.set_me(new_me = True)
        print("Welcome to Louis's Film Recommender!")
        print("Done initializing recommender!")
        #initiate old predictions
        with open("old_predictions.pickle","rb") as f:
                 cur_predictions = pickle.load(f)
        f.close()
        self.all_predictions = cur_predictions
        print('Done setting recommendations!')
        print("Type test.examples() for help")
        
    def set_me(self, new_me = False):
        if new_me:
            self.isnew = False
            self.random_rated = False
            self.my_recommend = False
            self.my_id = 672
        else:
            try:
                with open("my_setting.pickle","rb") as f:
                    self.isnew, self.random_rated, self.my_recommend, self.my_id = pickle.load(f)
                f.close()
            except:
                self.isnew = False
                self.random_rated = False
                self.my_recommend = False
                self.my_id = len(self.user_all_id) + 1
        self.save_me()
        
    def save_me(self):
        '''
        Function to save user's information.
        '''
        with open("my_setting.pickle","wb") as f:
            pickle.dump([self.isnew, self.random_rated, self.my_recommend, self.my_id], f, protocol=2)
        f.close()
        
    def read_data(self):
        links = pd.read_csv('links_small.csv')
        links.dropna(axis = 0, inplace = True)
        self.links = links
        self.film_data = pd.read_csv("film_merged.csv")
        self.idf = pd.read_csv("idf.csv")
        self.content_id = set(self.film_data.movieId)
        
    def set_user_rating(self):
        try:
            self.user_rating = pd.read_csv("new_user_rating.csv")
            self.path = "new_user_rating.csv"
        except:
            self.user_rating = pd.read_csv("ratings_small.csv")
            self.path = "ratings_small.csv"
            
        self.user_all_id = list(set(self.user_rating.userId))
        
    def random_user(self, rerank_ = True, n_reco = 10, show_rating = False):
        '''
        Function to randomly retrieve a user's recommendation.
        
        Parameters:
            
        rerank_: True or False
            Decide whether to rerank the result or not.
            Default is True.
        n_reco: From 10 - 90. Should be integer.
            Decide how many results you want.
            Default is 10.
        show_rating: True or False.
            Decide whether or not to show the user's all ratings.
            Default is False.
        
        '''
        num = np.random.randint(1, (len(self.user_all_id)+1))
        self.recommend_user(num, rerank_, n_reco, show_rating)
    #set all recommendations based on previous calculations using knn sigweight and hybird models
    def set_recommendations(self):
        try:
            with open("new_predictions.pickle","rb") as f:
                cur_predictions = pickle.load(f)
            f.close()
            self.all_predictions = cur_predictions
        except:
             with open("old_predictions.pickle","rb") as f:
                 cur_predictions = pickle.load(f)
             f.close()
             self.all_predictions = cur_predictions
        print('Done setting recommendations!')
    #function to retrieve recommendations
    def recommend_getter(self, user_id):
        if user_id > 671:
            cur_pred = self.my_predictions
        else:
            cur_pred = self.all_predictions
        user_pred = cur_pred[str(user_id)]
        if len(user_pred) == 0:
            return("NO.{} user not found!!".format(user_id))
        user_pred = pd.DataFrame(user_pred, columns = ['movieId','pred_rating'])
        user_pred.movieId = user_pred.movieId.astype('int64')
        print("Done retrieving NO.{} user's recommendations!!".format(user_id))
        return user_pred
    #check if the input is valid
    def check_var(self, user_id, rerank_, n_reco, show_rating):
        if isinstance(n_reco, int) == False:
            return("{} is not a integer! Please re-enter number smaller than 90.".format(n_reco))
        if n_reco > 90:
            return("{} is too big, please re-enter number smaller than 90".format(n_reco))
        if n_reco < 10:
            return("{} is too small, please re-enter number bigger than 10".format(n_reco))
        if isinstance(user_id, int) == False:
            return("Please enter a integer as user_id")
        if isinstance(rerank_, bool) == False:
            return("{} is not a boolean type. Enter True or Flase.".format(rerank_))
        if isinstance(show_rating, bool) == False:
            return("{} is not a boolean type. Enter True or Flase.".format(show_rating))
        return ("OK!")
    #main function for user to retrieve recommendations according tp user_id and they
    #can decide whether or not to rerank the result and the amount of recommendations n_reco
    def recommend_user(self, user_id = 10, rerank_ = True, n_reco = 10, show_rating = False):
        '''
        This is the function to get a specific user's recommendation.
        
        Parameters:
        
        user_id: From 1 to 671. Should be integer.
            If you have run recommend_me, you can use your own user_id.
            Default is 10.
        rerank_: True or False
            Decide whether to rerank the result or not.
            Default is True.
        n_reco: From 10 - 90. Should be integer.
            Decide how many results you want.
            Default is 10.
        show_rating: True or False.
            Decide whether or not to show the user's all ratings.
            Default is False.
        '''
        safe_check = self.check_var(user_id, rerank_, n_reco, show_rating)
        if safe_check != "OK!":
            print(safe_check)
            return
        #make sure the user rating file is the latest one
        self.set_user_rating()
        #get recommendations from saved file
        user_pred = self.recommend_getter(user_id)
        #do rerank
        if rerank_:
            user_pred, good_idx = rerank_1.rerank(user_id, self.user_rating, self.idf, user_pred, self.film_data, self.content_id, show_rating)
            if len(good_idx[0]) >= n_reco:
                user_pred = user_pred.iloc[good_idx[0]]
            else:
                need = n_reco - len(good_idx[0])
                pred1 = user_pred.iloc[good_idx[0]]
                retrieved = list(pred1.movieId)
                rat = user_pred.values.tolist()
                rat.sort(key=lambda x: x[1], reverse=True)
                for i in range(len(rat)):
                    if need == 0:
                        break
                    if rat[i][0] in retrieved:
                        continue
                    retrieved.append(rat[i][0]) 
                    need -= 1
                user_pred = pd.DataFrame(retrieved, columns = ['movieId'])
                user_pred.movieId = user_pred.movieId.astype('int64')
        #map titles according to tmdbId
        result = self.map_titles(user_pred)
        if rerank_:
            print("-------------Recommendations with reranking-------------")
        else:
            print("-------------Recommendations without reranking-------------")
        for idx, row in result.iterrows():
            if idx == n_reco:
                break
            print('Title:',row.title,'\t','Director:',row.directors)
            print('IMDB Rating:',row.vote_average,'\n')
            
    def map_titles(self, user_pred):
        mapping = self.film_data.copy()[['movieId','title','vote_average','directors']]
        result = pd.merge(user_pred, mapping)
        return result
    
    
    
    def random_film(self, novelty = 1):
        cur_data = self.film_data[['movieId','title','popularity','directors']]
        cur_list = cur_data.values.tolist()
        cur_list.sort(key=lambda x: x[2], reverse=True)
        new_list = random.sample(cur_list[:(300 * novelty)], 150 * novelty)
        list_data = pd.DataFrame(new_list, columns = ['movieId','title','popularity','directors'])[['movieId','title','directors']]
        
        return list_data
    
    def rate_for_fun(self, novelty = 1):
        '''
        Function to rate films randomly.
        
        Parameter:
        
        Novelty: From 1- 10. Should be integer.
            The degree of the film's novelty you want. Bigger number means more surprise!
        '''
        if int(novelty) < 1 or int(novelty) > 10:
            print("{} is not a valid novelty measure".format(novelty))
            return
        
        candidate_films = self.random_film(int(novelty))
        added = []
        print("Attention: Enter -1 to skip this film. If you want to end and save the ratings, enter 11. If you don't want to save, enter 12 or close the program.")
        print("——" * 20)
        print("\n")
        for index, row in candidate_films.iterrows():
            print(row.title,'\n',row.directors)
            s = input("Give your ratings:")
            print("——" * 20)
            s = s.split()[0]
            try:
                rat = float(s)
                
                if rat == -1:
                    continue
                if rat == 12:
                    return
                if rat == 11:
                    self.create_new(added, self.my_id)
                    self.random_rated = True
                    return
                if rat < 1 or rat > 10:
                    print("{} should be 1~5.".format(rat))
                    continue
                    #return
                added.append([self.my_id, row.movieId, rat])
            except:
                print("{} is not a valid rating".format(s))
                continue
            print("We have recorded {} entries data. Keep rating!".format(len(added)),'\n')
            
        print("Game over!")
        s = input("Enter 11 to save your data. Otherwise enter 12.")
        s = s.split()[0]
        if int(s) == 11:
            self.create_new(added, self.my_id)
            self.random_rated = True
        self.save_me()
        return
    def random_rate(self, candidate_films):
        '''
        Function to create profile for cold starters.
        '''
        added = []
        print("Attention: Enter -1 to skip this film. Enter -2 to exit and you need to start over.15 ratings needed.")
        print("——" * 20)
        print("\n")
        for index, row in candidate_films.iterrows():
            if len(added) > 14:
                break
            print(row.title,'\n',row.directors)
            s = input("Give your ratings:")
            print("\n\n")
            s = s.split()[0]
            try:
                rat = float(s)
                
                if rat == -1:
                    continue
                if rat == -2:
                    return
                
                if rat < 1 or rat > 5:
                    print("{} should be 1~5.".format(rat))
                    continue
                    #return
                added.append([self.my_id, row.movieId, rat])
            except:
                print("{} is not a valid rating".format(s))
                break
            print("We have recorded {} entries data. Keep rating!".format(len(added)))
                #return
        if len(added)  < 5:
            print("WOW! You ran out my tests! You must have a very special taste in films! Sorry, I can't give you any good advice.")
            return []
        if len(added) < 14:
            flag = input("If you still want to get recommendations (might be bad recommendations), press 1 other wise -1.")
            if int(flag) == -1:
                return []
        self.random_rated = True
        return added
    def recommend_me(self, rerank_ = True, n_reco = 10, show_rating = False):
        '''
        This is the function to get a your recommendations.
        
        Parameters:

        rerank_: True or False
            Decide whether to rerank the result or not.
            Default is True.
        n_reco: From 10 - 90. Should be integer.
            Decide how many results you want.
            Default is 10.
        show_rating: True or False.
            Decide whether or not to show the user's all ratings.
            Default is False.
        '''
        if self.isnew == False:
            #if never recommend before, then predict all!
            if self.my_recommend == False:
                candidate_films = self.random_film()
                print("Your user id is {}".format(self.my_id))
                added = self.random_rate(candidate_films)
                if self.random_rated == False:
                    print("Not enough data entries. You can try next time!")
                    return
                self.create_new(added, self.my_id)
                print("Calculating.........................It might take 2~5 mins depending on your computer")
                default_recommender.predict("new_user_rating.csv",self.my_id)
                print("Almost there!!!!!! Be patient!!!!")
        #if the data is new, we have to recalculate
        else:
            print("Calculating.........................It might take 2~5 mins depending on your computer")
            default_recommender.predict("new_user_rating.csv",self.my_id)
            print("Almost there!!!!!! Be patient!!!!")
        
        with open("my_predictions.pickle","rb") as f:
                my_predictions = pickle.load(f)
        f.close()
        self.my_predictions = my_predictions
        self.recommend_user(self.my_id, rerank_, n_reco, show_rating)
        self.my_recommend = True
        self.isnew = False
        self.save_me()
    
    def update_all(self):
        '''
        Function to update all users' recommendations.
        '''
        print("Calculating.........................It might take 10~15 mins depending on your computer")
        default_recommender.predict("new_user_rating.csv", predict_all = True)
        
        self.isnew = False
        self.my_recommend = True
        self.save_me()
    #create new user rating file after the user has rated new films
    def create_new(self, added, my_id):
        test = pd.DataFrame(added, columns = ['userId','movieId','rating'])
        #make sure the user rating is the latest one
        self.set_user_rating()
        self.new_user_rating = pd.concat([self.user_rating, test])
        self.new_user_rating.to_csv("new_user_rating.csv", index = None)
        print("Done saving your profile") 
        #update user rating
        self.set_user_rating()
        #mark we have new data now!
        self.isnew = True
        self.save_me()
        
        
    def rate(self, user_id = 672):
        '''
        Function to rate new films.
        
        Parameters:
        
        user_id: Your user id. Default is 672. Should be greater than 671 since smaller number are original users, you should't update their ratings.
        '''
        my_id = int(user_id)
        print("Your user id is {}".format(my_id))
        if my_id < 672:
            print("You are a new user. Please enter id greater than 671. The default id is 672.")
            #return
        flag = 1
        added = []
        cur_data = self.film_data.copy()
        while flag == 1:
            imdb = input("What's the imdb id of the film?Please don't enter tt in the imdb id.")
            imdb = int(imdb.lstrip("0"))
            try:
                cur_film = cur_data.iloc[cur_data.index[cur_data.imdbId == imdb]]
            except:
                print("Something wrong!")
                flag = int(input("Do you want to try again? Press 1 to retry. Otherwise press -1."))
                continue
            if len(cur_film) == 0:
                print("This film is not in our database. Contact me to add it!")
                flag = int(input("Do you want to try again? Press 1 to retry. Otherwise press -1."))
                continue
            print("The film is {}".format(cur_film.title.values[0]))
            rating = input("Your rating is:")
            try:
                rat = float(rating)
                if rat < 1 or rat > 5:
                    print("{} should be 1~5.".format(rat))
                    continue
                    #return
                added.append([my_id, cur_film.movieId.values[0], rat])
            except:
                print("{} is not a valid rating".format(rat))
                continue
            flag = int(input("Do you want to continue? Press 1 to continue. Otherwise press -1."))
        if len(added) == 0:
            return
        self.create_new(added, my_id)
        print("Your data has saved and we have updated user rating file.")
    def baseline_recommender_getter(self, user_id, model):
        with open("baseline_predictions.pickle","rb") as f:
               bsl_pred, svd_pred = pickle.load(f)
        f.close()
        if model == "baseline":
            user_pred = bsl_pred[str(user_id)]
        elif model == 'svd':
            user_pred = svd_pred[str(user_id)]
        else:
            print("{} model not found. We only calculate baseline and svd model. Please enter baseline or svd.".format(model))
        if len(user_pred) == 0:
            return("NO.{} user not found!!".format(user_id))
        user_pred = pd.DataFrame(user_pred, columns = ['movieId','pred_rating'])
        user_pred.movieId = user_pred.movieId.astype('int64')
        print("Done retrieving NO.{} user's recommendations!!".format(user_id))
        return user_pred
    
    def bsl_recommend_user(self, user_id = 10, n_reco = 10, model = "baseline"):
        '''
        This is the function to get a specific user's baseline model recommendation.
        
        Parameters:
        
        user_id: From 1 to 671. Should be integer.
            If you have run recommend_me, you can use your own user_id.
            Default is 10.
        n_reco: From 10 - 90. Should be integer.
            Decide how many results you want.
            Default is 10.
        model: baseline or svd. Should be string format.
            The model you want to compare.
        '''
        user_pred = self.baseline_recommender_getter(user_id, model)
        result = self.map_titles(user_pred)
        print("------------{} Model Recommendations------------------".format(model))
        for idx, row in result.iterrows():
            if idx == n_reco:
                break
            print('Title:',row.title,'\t','Director:',row.directors)
            print('IMDB Rating:',row.vote_average,'\n')
            
    def bsl_recommend(self):
        '''
        Function to calculate recommendations. Use it if you add new data.
        '''
        self.set_user_rating()
        bsl_recommender.predict(self.path)
                    
        
      
    def examples(self):
        '''
        Function to print a introduction of all functions user can use.
        '''
        example = open("examples.txt", 'r')
        print(example.read())
