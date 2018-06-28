# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:36:59 2018

@author: Anagha Kelkar
"""

# dataset
# userId | movieId | rating | timestamp

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print training and testing data
print("Training dataset: ",repr(data['train']))
print("Testing dataset: ",repr(data['test']))

# ------------- Create Model ---------------
# weighted approximate rank pairwise - Gradient descent algorithm
# (past + reading history) content + collaborative (similar users rating) = hybrid
model = LightFM(loss='warp') 

# ------------- Train Model ----------------
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
    
    # generate recommendation for each user as input
    for user_id in user_ids:
        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        
        # print out the results
        print("User %s" % user_id)
        print("     Known positives:")
        
        for x in known_positives[:3]:
            print("          %s" % x)
        
        print("     Recommended:")
        
        for x in top_items[:3]:
            print("          %s" % x)

# take raw inputs from command line
input = []
no_of_users = raw_input("Enter number of users \n")
for i in range(int(no_of_users)):
    j = i+1
    user_id = raw_input("Enter id of user %s \n" % j)
    input.append(user_id)
sample_recommendation(model, data, input)