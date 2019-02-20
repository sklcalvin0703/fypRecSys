from flask import Flask,jsonify
from Recommender import loadDataSet,favoriteMovies
app = Flask(__name__)

@app.route('/favoriteMovies')
def index():
    data, moviedata = loadDataSet()
    list = favoriteMovies(3,3,data)
    return jsonify(list)


if __name__ == '__main__':
    app.run(debug=True)