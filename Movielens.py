import numpy as np
import pandas as pd
import os

class Movielens:
    def __init__(self):
        self.dir = os.getcwd()
        self.ratingsPath = self.dir + '/ml-latest-small/ratings.csv'
        self.moviesPath = self.dir + '/ml-latest-small/movies.csv'

    def loadDataSet(self):
        movieData = pd.read_csv(self.moviesPath,header=None,skiprows=1, 
                 names=['movieId', 'title', 'genres'])

        ratingData = pd.read_csv(self.ratingsPath,header=None,skiprows=1, 
                 names=['userId', 'movieId', 'rating', 'timestamp'])
        ratingData = ratingData.drop(['timestamp'],axis=1)

        return movieData, ratingData