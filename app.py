import os
from flask import Flask,jsonify
from Recommender import loadDataSet,favoriteMovies
app = Flask(__name__)

@app.route('/favoriteMovies')
def index():
    data, moviedata, userRatingMatrix = loadDataSet()
    list = favoriteMovies(3,3,data)
    print(os.getcwd())
    return jsonify(list)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port = 5000)