import os
import csv
import sys
import re
import pandas as pd
import numpy as np
import random
from random import randint
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from sklearn.externals import joblib
from surprise import Dataset
from surprise import Reader
from surprise import SVD


class Environment:
    def __init__(self, userID, action):
        self.userID = userID
        self.userHistory = userHistory
        self.SVD = self.load_model()
    
    def step(self, action):
        prediction = self.SVD.predict(self.userID, self)
        estimatedRating = prediction[3]
        
        reward = estimatedRating
        self.selectedUser[0,action] = reward #update the user's state(history)
        next_state = self.selectedUser
        
        return next_state, reward 

    def load_model(self):
        joblib.load("./SVDtuned.pkl")