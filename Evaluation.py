from Movielens import Movielens
from surprise import KNNBasic
from surprise import Reader
from surprise import SVD, Dataset
from surprise import accuracy
from surprise.model_selection import LeaveOneOut, train_test_split
import numpy as np
import matplotlib.pyplot as plt

ml =  Movielens()

movieData, ratingData = ml.loadDataSet()

reader = Reader()
data = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)
TrainData, TestSet = train_test_split(data, test_size=.25, random_state=1)

results = {}
#userknn cosine
userKNN_cosine = KNNBasic(sim_options={'name': 'cosine', 'user-based': True})
userKNN_cosine.fit(TrainData)
userKNN_msd = KNNBasic(sim_options={'name': 'msd', 'user-based': True})
userKNN_msd.fit(TrainData)
userKNN_pearson = KNNBasic(sim_options={'name': 'pearson', 'user-based': True})
userKNN_pearson.fit(TrainData)

predictions = userKNN_cosine.test(TestSet)
print("On user KNN with cosine similarity metrics : ")
userKNN_cosine_rmse = accuracy.rmse(predictions, verbose=False)
userKNN_cosine_mae = accuracy.rmse(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))
print("\n")

#userknn msd
predictions = userKNN_msd.test(TestSet)
print("On user KNN with msd similarity metrics : ")
userKNN_msd_rmse = accuracy.rmse(predictions, verbose=False)
userKNN_msd_mae = accuracy.mae(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))
print("\n")

#userknn pearson
predictions = userKNN_pearson.test(TestSet)
print("On user KNN with pearson similarity metrics : ")
userKNN_pearson_rmse = accuracy.rmse(predictions, verbose=False)
userKNN_pearson_mae = accuracy.mae(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))
print("\n")

#itemknn
itemKNN_cosine = KNNBasic(sim_options={'name': 'cosine', 'user-based': False})
itemKNN_cosine.fit(TrainData)
itemKNN_msd = KNNBasic(sim_options={'name': 'msd', 'user-based': False})
itemKNN_msd.fit(TrainData)
itemKNN_pearson = KNNBasic(sim_options={'name': 'pearson', 'user-based': False})
itemKNN_pearson.fit(TrainData)

predictions = itemKNN_cosine.test(TestSet)
print("On item KNN cosine: ")
itemKNN_cosine_rmse = accuracy.rmse(predictions, verbose=False)
itemKNN_cosine_mae = accuracy.mae(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))
print("\n")

predictions = itemKNN_msd.test(TestSet)
print("On item KNN msd: ")
itemKNN_msd_rmse = accuracy.rmse(predictions, verbose=False)
itemKNN_msd_mae = accuracy.mae(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))
print("\n")
svd = SVD()

predictions = itemKNN_pearson.test(TestSet)
print("On item KNN pearson: ")
itemKNN_pearson_rmse = accuracy.rmse(predictions, verbose=False)
itemKNN_pearson_mae = accuracy.mae(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))
print("\n")

svd.fit(TrainData)
predictions = svd.test(TestSet)
print("On SVD: ")
svd_rmse = accuracy.rmse(predictions, verbose=False)
svd_mae = accuracy.mae(predictions, verbose=False)
print("RMSE: ", accuracy.rmse(predictions, verbose=False))
print("MAE: ", accuracy.mae(predictions, verbose=False))


n_groups = 7
rmse_scores = [userKNN_cosine_rmse, userKNN_msd_rmse, userKNN_pearson_rmse, itemKNN_cosine_rmse, itemKNN_msd_rmse, itemKNN_pearson_rmse, svd_rmse]
mae_scores = [userKNN_cosine_mae, userKNN_msd_mae, userKNN_pearson_mae, itemKNN_cosine_mae, itemKNN_msd_mae, itemKNN_pearson_mae]

objects = ('userKNN-cosine', 'userKNN-msd', 'userKNN-pearson', 'itemKNN-cosine', 'itemKNN-msd', 'itemKNN-pearson', 'SVD')
index = np.arange(len(objects))

bar_width = 0.35
opacity = 0.8

plt.bar(index, rmse_scores, align='center', alpha=0.5)
 
# rects2 = plt.bar(index + bar_width, mae_scores, bar_width,
# alpha=opacity,
# color='g',
# label='MAE')

plt.xlabel('Algorithms')
plt.ylabel('Scores')
plt.title('RMSE score by different algoritms')
plt.xticks(index, objects)

plt.tight_layout()
plt.show()