import numpy as np
import pandas as pd
import os

#generate movie recommednation, given movie they have already watched,
#and the ratings they gave for those movies

#pandas, a data analysis library, for most of the data preparation
#and analysis. We cna read the data from a csw, write to a csv,
#manipulate it into different shapres

def loadDataSet():
    dir = os.getcwd()
    dataFile = dir + '/ml-100k/u.data'
    data=pd.read_csv(dataFile, sep="\t", header=None, 
                 names=['userId', 'itemId', 'rating', 'timestamp'])
    movieInfoFile  = dir + '/ml-100k/u.item'
    moviedata = pd.read_csv(movieInfoFile,sep="|", header=None,index_col=False,names=['itemId','title'],usecols=[0,1],encoding='latin-1')
    data=pd.merge(data,moviedata,left_on='itemId',right_on="itemId")


    data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])

    userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                   index=['userId'], columns=['itemId'])

    return data,moviedata, userItemRatingMatrix


def favoriteMovies(activeUser, N, data):
    #1 subset the dataframe to have the rows corresponding to the active user
    #2 sort by rating  in descending order
    topMovies=pd.DataFrame.sort_values(
        data[data.userId==activeUser], ['rating'],ascending=[0])[:N]

    return  list(topMovies.title)


# from scipy.spatial.distance import correlation
# def similarity(user1,user2):
#     user1=np.array(user1)-np.nanmean(user1)  #nanmean -> calculating the  mean ignoring the NaN
#     user2=np.array(user2)-np.nanmean(user2)
    
#     #movie have in common
#     commonItemIds = [i for i in range(len(user1))
#                                       if user1[i]>0 and user2[i]>0]
    
#     if len(commonItemIds)==0:
#         return 0
#     else:
#         user1=np.array([user1[i] for i in commonItemIds])
#         user2=np.array([user2[i] for i in commonItemIds])
#         return correlation(user1,user2)

# def nearestNeighbourRatings(activeUser, K):
#     similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,
#                                  columns=['Similarity'])
    
#     for i in userItemRatingMatrix.index:
#         similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
#                                           userItemRatingMatrix.loc[i])
#         #find the similarting and store into the similarity matrix
        
#     similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
#     #sort in descending order
    
#     #Nearest K neighbours
#     nearestNeighbours=similarityMatrix[:K]
    
#     #now predict
#     neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    
#     # a placeholder for the predicted item ratings
#     predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
    
#     for i in userItemRatingMatrix.columns:
#         predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
        
#         for j in neighbourItemRatings.index:
#             if userItemRatingMatrix.loc[j,i]>0:
                
#                 predictedRating += (userItemRatingMatrix.loc[j,i]
#                                     -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
#         predictItemRating.loc[i,'Rating']=predictedRating
#     return predictItemRating
    
    
# def topNRecommendation(activeUser, N):
#     predictItemRating=nearestNeighbourRatings(activeUser, 10)
#     movieAlreadyWatched=list(userItemRatingMatrix.loc[activeUser].loc[userItemRatingMatrix.loc[activeUser]>0].index)
#     predictItemsRating=predictItemRating.drop(movieAlreadyWatched)
#     print(predictItemsRating)
#     topRecommendations=pd.DataFrame.sort_values(predictItemRating,
#                                                ['Rating'],ascending=[0])[:N]
#     topRecommendationTitles=(moviedata.loc[moviedata.itemId.isin(topRecommendations.index)])
#     return list(topRecommendationTitles.title)
