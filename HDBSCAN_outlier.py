import pandas as pd
import numpy as np

from find_elbow import *

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import hdbscan

from sklearn.metrics import davies_bouldin_score

def hdbscan_outlier(X,n_neighbors=3):

    # instantiate nearest neighbors model and get distances
    neigh = NearestNeighbors(n_neighbors= n_neighbors, metric = 'euclidean')

    neigh.fit(X)

    distances, indices = neigh.kneighbors(X)

    # isolate the unique distances of the nth nearest neighbor
    distances = np.sort(distances[:,-1])

    # look for the inflection point
    lst_eps = find_elbow(distances,cv='convex')
#     print(lst_eps)

    lst_db_score = []
    lst_adj_score = []
    lst_tup_param = []

    for eps in lst_eps:


        for method in 'eom leaf'.split():

            # try:

            ## fit HDBSCAN model to identify outliers
            clusterer = hdbscan.HDBSCAN(min_cluster_size=n_neighbors,
                                        cluster_selection_epsilon=eps,
                                        cluster_selection_method = method)
            clusterer.fit(X)

            lst_tup_param.append((eps,method))

            lst_db_score.append(davies_bouldin_score(X, clusterer.labels_))

            # except:
            #     pass


    params = lst_tup_param[np.argmin(lst_db_score)]

    eps = params[0]
    method = params[1]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=n_neighbors,
                                cluster_selection_epsilon=eps,
                                cluster_selection_method = method)
    clusterer.fit(X)

    # identify outliers
    arr_out = X[np.argwhere(clusterer.labels_==-1)]
    arr_in = X[np.argwhere(clusterer.labels_!=-1)]
    
    return arr_out,arr_in,clusterer
