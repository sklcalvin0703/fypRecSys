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

class Agent:
    def __init__(self, model):
        self.memory = deque(maxlen=2000) #use it for storing history
        self.GAMMA = 0.9
        self.EPSILON = 0.4
        self.learning_rate = 0.001 #for model
        self.model = self.load_model()

    def load_model():
        return joblib.load("./DQNmodel.pkl")

    def remember_memory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
  
    def make_act(self, state):
        if np.random.rand() <= EPSILON: #exploration
            restart = True
            while restart:
                restart = False
                action = random.randrange(self.action_size)
                for i in range(len(self.memory)):
                    if self.memory[i][1] == action:
                        restart = True
                        break             
            return action
        action_values = self.model.predict(state)
        return np.argmax(action_values[0]) #exploitation
    
    def experience_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    