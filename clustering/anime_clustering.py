import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# from mpl_toolkits import mplot3d


dataAnime = pd.read_csv("anime.csv")
# clean the data
dataAnime = dataAnime.set_index('anime_id').dropna()
dataAnime = dataAnime[dataAnime['type'] == 'TV']
dataAnime = dataAnime[dataAnime['episodes'] != 'Unknown']
dataAnime['episodes'] = pd.to_numeric(dataAnime['episodes'])

# shounen_data = data[data['genre'].str.contains("Shounen")]
subsetAnimeData = dataAnime
subsetAnimeData = subsetAnimeData[['episodes', 'members', 'rating']]
# Check mean/variance
animeStats = pd.DataFrame({"Mean": subsetAnimeData.mean(), "Var": subsetAnimeData.var()})
print(animeStats)

# standardize animeData
columns = subsetAnimeData.columns.to_list()
standardizedSubsetAnimeData = pd.DataFrame(columns=columns, index=subsetAnimeData.index)
standardizedSubsetAnimeData.index.name = 'anime_id'
standardization = StandardScaler().fit_transform(subsetAnimeData)
standardizedSubsetAnimeData[['episodes', 'members', 'rating']] = standardization

def kmeansOnAnimeData(data=standardizedSubsetAnimeData, clusters=2):
    # print(clusters)
    km = KMeans(n_clusters=clusters, random_state=10)
    prediction = km.fit_predict(data)
    plotClusters(prediction, data, clusters)
    plotClusters(prediction, subsetAnimeData, clusters)
    return prediction

def gmOnAnimeData(data=standardizedSubsetAnimeData, clusters=2):
    gm = GaussianMixture(n_components=clusters, random_state=10)
    predictionGM = gm.fit_predict(data)
    plotClusters(prediction=predictionGM, data=data, clusters=clusters)
    # plotClusters(prediction=predictionGM, clusters=clusters)
    return predictionGM

# Inspired by: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb#:~:text=on%20this%20dataset.-,The%20Elbow%20Method,becomes%20first%20starts%20to%20diminish.
def checkSilouhette(data):
    kmax = 10
    sil = []
    for k in range(2, kmax + 1):
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

checkSilouhette(standardizedSubsetAnimeData) # Best with 3 clusters
prediction = kmeansOnAnimeData(clusters=3)
# subsetAnimeData['label'] = prediction
# subsetAnimeData.to_csv('anime_w_label.csv')
# gmOnData()

