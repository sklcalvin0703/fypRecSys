import os
import sys
import json
import pandas as pd
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from sklearn.externals import joblib
from surprise import Reader, Dataset, SVD
from flask import Flask,jsonify,request
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
app = Flask(__name__)

@app.route('/', methods=["GET"])
def testing():
    return "<h1>{{test}}</h1>"

@app.route('/SVDrecommender', methods=["POST"])
def SVDrecommender():
    # userId = request.json['userId']
    # print(request.json['userId'])
    #preprocessing
    #print(request.json)
    training_set = pd.read_pickle("./training_data.pkl")
    training_set['userId'] = training_set['userId'].apply(str)
    tempuserId = request.json['userId']
    tempdata = request.json['data']
    temp = []
    print(training_set.dtypes)
    print(training_set.head())
    print(tempdata)
    for data in tempdata:
        print(data['movieId'])
        temp.append([tempuserId, int(data['movieId']), float(data['rating'])])

    
    tempdf = pd.DataFrame(temp, columns=['userId','movieId', 'rating'])

    newTrainData = training_set.append(tempdf, ignore_index=True)


    os.remove("./training_data.pkl")
    print('Data before drop duplicates:')
    print(newTrainData.tail())
    print('Data after drop duplicates:')
    newTrainData = newTrainData.drop_duplicates(subset=['userId','movieId'])
    print(newTrainData.tail())

    #save new data
    newTrainData.to_pickle("training_data.pkl")

    #print(newTrainData.loc[newTrainData['userId'] == tempuserId])
    
    reader = Reader()
    newDataSet = Dataset.load_from_df(newTrainData[['userId', 'movieId', 'rating']], reader)
    newDataSet = newDataSet.build_full_trainset()

    model = joblib.load("./SVDtuned.pkl")
    print('Training in progress:')
    model.fit(newDataSet)

    #remove old model and data
    joblib.dump(model, "SVDtuned.pkl")

    
    recommendations  = makerecommendation(model,newDataSet,tempuserId)[:10]
    print(recommendations)
    
    recommendedMovieId = []
    for item in recommendations:
        recommendedMovieId.append(item[0])
    
    return jsonify(recommendedMovieId)


def makerecommendation(model, newdata, userId):
    recommendation = []
    the_iid_list = newdata.all_items()
    for iid in the_iid_list:
        prediction = model.predict(userId, iid)
        intMovieId = int(prediction[1])
        estimatedRating = prediction[3]
        recommendation.append((intMovieId,estimatedRating))
    
    recommendation.sort(key=lambda x:x[1], reverse=True)

    return recommendation

@app.route('/DQNrecommender', methods=["POST"])
def DQNrecommender():
    #print(request.json)
    newTrainData, tempuserId = datapreprocessing(request.json)

    movie_data = pd.read_pickle("./movie_data.pkl")
    user_item_matrix = combine_matrix(movie_data, newTrainData).fillna(0)
    user_item_matrix.to_pickle("test_data.pkl") #debug purpose

    #get current used rating matrix (state)
    userENV = user_item_matrix.loc[[tempuserId]].to_numpy()
    print("This is USER history:")
    print(userENV)
    DQNmodel = joblib.load("./DQNmodel.pkl")

    #always take greedy action
    rawID_prediction  = DQNmodel.predict(userENV)
    #sort top 10 result
    rawID_prediction = rawID_prediction[0].argsort()[-10:][::-1]
    realID_prediction = []
    #transfer raw id to real movie id
    for id in rawID_prediction:
        temp = int(movie_data.iloc[id]['movieId'])
        realID_prediction.append(temp)
    
    print(realID_prediction)
    return jsonify(realID_prediction)

def combine_matrix(movie, rating):
    merged = pd.merge(rating,movie,left_on='movieId',right_on="movieId")
    return pd.pivot_table(merged, values='rating',index=['userId'], columns=['movieId'])

def datapreprocessing(request):
    training_set = pd.read_pickle("./training_data.pkl")
    training_set['userId'] = training_set['userId'].apply(str)
    tempuserId = request['userId']
    tempdata = request['data']
    temp = []
    print(training_set.dtypes)
    print(training_set.head())
    #print(tempdata)
    for data in tempdata:
        #print(data['movieId'])
        temp.append([tempuserId, int(data['movieId']), float(data['rating'])])
    tempdf = pd.DataFrame(temp, columns=['userId','movieId', 'rating'])

    newTrainData = training_set.append(tempdf, ignore_index=True)

    os.remove("./training_data.pkl")
    print('Data before drop duplicates:')
    print(newTrainData.tail())
    print('Data after drop duplicates:')
    newTrainData = newTrainData.drop_duplicates(subset=['userId','movieId'])
    print(newTrainData.tail())
    #save new data
    newTrainData.to_pickle("training_data.pkl")
    return newTrainData, tempuserId




if __name__ == '__main__':
    os.environ['KERAS_BACKEND'] = 'Theano'
    app.run(host='0.0.0.0', port = 5000)