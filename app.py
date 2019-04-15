import os
import sys
import json
import pandas as pd
from sklearn.externals import joblib
from surprise import Reader, Dataset, SVD
from flask import Flask,jsonify,request

app = Flask(__name__)

@app.route('/', methods=["GET"])
def testing():
    return "<h1>{{test}}</h1>"

@app.route('/SVDrecommender', methods=["POST"])
def SVDrecommender():
    # userId = request.json['userId']
    # print(request.json['userId'])

    #preprocessing
    print(request.json)
    training_set = pd.read_pickle("./training_data.pkl")
    training_set['userId'] = training_set['userId'].apply(str)
    tempuserId = request.json['userId']
    tempdata = request.json['data']
    temp = []
    print(training_set.dtypes)
    print(training_set.head())
    print(tempdata)
    for data in tempdata:
        print('hi test')
        print(data['movieId'])
        temp.append([tempuserId, int(data['movieId']), float(data['rating'])])

    print('im temp')
    print(temp)
    
    tempdf = pd.DataFrame(temp, columns=['userId','movieId', 'rating'])
    #print(tempdf.dtypes)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)