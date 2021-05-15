import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from mpl_toolkits import mplot3d

anime_clusters = 3
user_clusters = 2
dataUser = pd.read_csv("rating.csv")
dataAnimePrediction = pd.read_csv("anime_w_label.csv")
# clean the data
dataUser = dataUser.dropna()
dataUser = dataUser[dataUser['rating'] != -1]
dataUser = dataUser.merge(dataAnimePrediction[['anime_id', 'label']], on='anime_id', how='inner')
meanRatingPerUserPerLabel = dataUser.groupby(['user_id', 'label'])['rating'].mean().to_frame().\
    rename(columns={'col_0': 'rating'}).\
    reset_index()
clusteringUserData = meanRatingPerUserPerLabel.pivot_table(values='rating', index='user_id', columns='label', fill_value=0).\
    rename(columns={0:'anime_label_0', 1:'anime_label_1', 2:'anime_label_2'})
print(clusteringUserData)

def kmeansOnUserData(data, clusters=2):
    # print(clusters)
    km = KMeans(n_clusters=clusters, random_state=10)
    prediction = km.fit_predict(data)
    plotClusters(prediction, data, clusters)
    return prediction

# Inspired by: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb#:~:text=on%20this%20dataset.-,The%20Elbow%20Method,becomes%20first%20starts%20to%20diminish.
def checkSilouhette(data):
    kmax = 10
    sil = []
    for k in range(2, kmax + 1):
        print(f'Staring {k}')
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))
    clusters_trial = [i for i in range(2, kmax + 1)]
    print(sil)
    plt.plot(clusters_trial, sil)
    plt.legend()
    plt.xlabel("K_index")
    plt.ylabel("silouhette_score")
    plt.show()

#3d becuase there are 3 anime clusters
def plotClusters(prediction, data, clusters=2):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_zlabel(data.columns[2])
    ax1.set_ylabel(data.columns[1])
    ax1.set_xlabel(data.columns[0])
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_zlabel(data.columns[2])
    ax2.set_ylabel(data.columns[1])
    ax2.set_xlabel(data.columns[0])
    for i in range(clusters):
        slice = data[prediction == i]
        ax2.scatter(slice.iloc[:, 0], slice.iloc[:, 1], slice.iloc[:, 2], label=i)
    plt.legend()
    plt.show()

# checkSilouhette(clusteringUserData) # Suggest user_clusters = 2
prediction = kmeansOnUserData(clusteringUserData)
clusteringUserData['prediction'] = prediction
# clusteringUserData.to_csv('user_w_label.csv')
# gmOnData()

