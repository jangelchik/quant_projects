import pandas as pd
import numpy as np

from scaler import *
from pca import *

import multiprocessing as mp

from typing import Union

from sklearn.metrics import silhouette_score,pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans, OPTICS

'''DBSCAN'''
def dbscan(df:pd.DataFrame,
           lst_wt:list=[None], 
           metric:str = 'euclidean',
           lst_mp:Union[list,None]=None):
    
    '''
    PARAMETERS:
    df: Pandas DataFrame - our feature set
    lst_wt: list - List of Pandas Series to weight our feature set variables, if None, no weighting is done
    metric: str - the distance metric we want to use
    lst_mp: list - integers you want to test for as min_pts parameters
    
    
    RETURNS:
    *** In a dictionary ***
    df_pca: Pandas DataFrame - PCA-Transformed feature set
    pca: Sklearn PCA object - fit PCA object
    df_scaled: Pandas DataFrame - scaled feature set
    scaler: Sklearn scaler object - Either Standard (z-score) or MinMax
    df_famd: Pandas DataFrame - the weights and mean values for transforming categorical features with FAMD
    s_wt: Pandas Series - series used to weight the feature set
    eps: float - epsilon value used for DBSCAN algorithm
    min_pts: int - minimum sample argument for DBSCAN algorithm
    labels: Pandas Series - Labels assigned to our data
    silhouette_score: float - Silhouette score of our DBSCAN model
    '''
    lst_d = [{'df_pca':df.copy()}]
    
    if metric != 'precomputed':
        # capture scaling iterables
        lst_iter = []

        # test both ZScore and MinMax scaling
        for scale_type in 'mm'.split():
            
            # make sure we perform yj transformation
            yj = True

            # try with and without FAMD
            for famd in [True,False]:

                for s_wt in lst_wt:

                    tup_iter = (df,scale_type,famd,s_wt,yj)

                    lst_iter.append(tup_iter)

    
        # Scale in parallel
        pool = mp.Pool(mp.cpu_count())

        results = pool.starmap_async(scale_df,lst_iter).get()

        #### append our results dictionaries to a list of iterables
        lst_iter = []

        for re in results:
            lst_iter.append(tuple(re.values()))

        ### Perform PCA in parallel
        pool = mp.Pool(mp.cpu_count())

        results = pool.starmap_async(pca,lst_iter).get()

        lst_d = []

        for re in results:
            lst_d.append(re)

    ### For all of our PCA-transformed data, perform DBSCAN clustering
    lst_d_fit = []

    for d_ in lst_d:

        df_pca = d_['df_pca']
        
        X = df_pca.copy()
        if metric != 'precomputed':
            X = X.drop_duplicates()

        # test various minpts values
        if not lst_mp:
            lst_mp = [int(np.log(X.shape[0]))+1,int(np.sqrt(X.shape[0]))+1,X.shape[1]*2]
        
        for minpts in lst_mp:

            if minpts > X.shape[0]:
                continue

            min_pts = minpts

            # instantiate nearest neighbors model and get distances
            neigh = NearestNeighbors(n_neighbors= min_pts , metric = metric)

            neigh.fit(X)

            distances, indices = neigh.kneighbors(X)

            # isolate the unique distances of the nth nearest neighbor
            distances = np.sort(distances[:,-1])

            # look for the inflection point
            lst_eps = find_elbow(distances,cv='convex')

            X_fit = df_pca.copy()

            '''USING EPS VALUES BASED ON UNIQUE ROWS, FIT MODEL WITH ALL ROWS (INCLUDING DUPLICATES)'''
            for eps in lst_eps:

                # here we'll make a dictionary to store all of our DBSCAN parameters
                d_fit  = d_.copy()
                d_fit['eps'] = eps
                d_fit['min_pts'] = min_pts

                # now fit the model
                db = DBSCAN(eps=eps, min_samples=min_pts, metric = metric, n_jobs = -1).fit(X_fit)
                
                d_fit['model'] = db

                db_labels = db.labels_+1
                db_labels[db_labels==0]=-1

                s_labels = pd.Series(db_labels,name='cluster').astype('str')
                s_labels.index = X_fit.index

                d_fit['labels'] = s_labels

                # get silhouette score of dbscan
                # decide if we want to require multiple clusters to count our eps iteration
                sil_dbscan = 0
                if s_labels.nunique() > 1:
                    sil_dbscan = silhouette_score(X_fit, s_labels)

                d_fit['silhouette_score'] = sil_dbscan

                lst_d_fit.append(d_fit)

    #### iterate through our results dictionaries, selecting the highest silhouette score
    d_best = lst_d_fit[0]

    for d_fit in lst_d_fit:

        if d_fit['silhouette_score'] > d_best['silhouette_score']:
            d_best = d_fit

    return d_best

'''KMeans'''
def kmeans(df:pd.DataFrame,
           lst_wt:list=[None],
           lst_k:Union[list,None]=None):
    
    '''
    PARAMETERS:
    df: Pandas DataFrame - our feature set
    lst_wt: list - List of Pandas Series to weight our feature set variables, if None, no weighting is done
    
    
    RETURNS:
    *** In a dictionary ***
    df_pca: Pandas DataFrame - PCA-Transformed feature set
    pca: Sklearn PCA object - fit PCA object
    df_scaled: Pandas DataFrame: scaled feature set
    scaler: Sklearn scaler object - Either Standard (z-score) or MinMax
    df_famd: Pandas DataFrame - the weights and mean values for transforming categorical features with FAMD
    s_wt: Pandas Series - series used to weight the feature set
    k: int - the number of clusters in our model
    km: Sklearn KMeans object - fit KMeans model
    labels: Pandas Series - Labels assigned to our data
    silhouette_score: float - Silhouette score of our DBSCAN model
    '''

    #### Scaling
    # capture scaling iterables
    lst_iter = []

     # test both ZScore and MinMax scaling
    for scale_type in 'std mm'.split():

        # make sure we perform yj transformation
        yj = True

        # try with and without FAMD
        for famd in [True,False]:

            for s_wt in lst_wt:

                tup_iter = (df,scale_type,famd,s_wt,yj)

                lst_iter.append(tup_iter)

    # iterate in parallel
    pool = mp.Pool(mp.cpu_count())

    results = pool.starmap_async(scale_df,lst_iter).get()

    #### append our results dictionaries to a list of iterables
    lst_iter = []

    for re in results:
        lst_iter.append(tuple(re.values()))

    #### PCA

    pool = mp.Pool(mp.cpu_count())

    results = pool.starmap_async(pca,lst_iter).get()

    lst_d = []

    for re in results:
        lst_d.append(re)

    #### KMeans iterations
    # capture our iterables
    lst_d_fit = []

    for d_ in lst_d:

        df_pca = d_['df_pca']

        X = df_pca.drop_duplicates()

        # generate k values to test based on size of feature set
        if not lst_k:
            lst_k = list(range(2,int(np.sqrt(X.shape[0]))+1))

        for idx,k in enumerate(lst_k):

            # can't have more clusters than datapoints 
            if k >= X.shape[0]:
                continue

            d_fit = d_.copy()

            # try fitting our model
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit model on our unique-rows-only DataFrame
                km = KMeans(n_clusters=k,n_init=10).fit(X)
                d_fit['model'] = km

                # now get our labels for the whole dataset
                km_labels = km.predict(df_pca)
                km_labels = km_labels + 1
                s_labels = pd.Series(km_labels,name='cluster').astype('str')
                s_labels.index = df_pca.index
                d_fit['labels'] = s_labels

                # get silhouette score of unique-rows model
                sil_km = silhouette_score(df_pca, km_labels)
                d_fit['silhouette_score'] = sil_km

                lst_d_fit.append(d_fit)

    #### iterate through our results dictionaries, selecting the highest silhouette score
    d_best = lst_d_fit[0]

    for d_fit in lst_d_fit:

        if d_fit['silhouette_score'] > d_best['silhouette_score']:
            d_best = d_fit

    return d_best

def transform(df:pd.DataFrame,
              scaler:Union[StandardScaler,MinMaxScaler],
              df_famd:pd.DataFrame,
              s_wt:pd.Series,
              pca:PCA):
    
    '''
    PARAMETERS:
    df: Pandas DataFrame - 2D DataFrame for either single or batch predictions
    model: the deserialized model loaded by DRUM or by `load_model`, if supplied
    
    RETURNS:
    df_ready: Pandas DataFrame - 2D DataFrame scaled and PCA-transformed, ready to be fed into our fit model
    '''
    
    '''READ IN OUR RELEVANT OBJECTS'''

    # read df_famd
    df_famd = pd.read_csv('df_famd.csv').set_index('params')

    # read s_wt
    s_wt = pd.read_csv('s_wt.csv').set_index('Feature').iloc[:,0]
    
    '''NOW PREPARE OUR DATA FOR THE FINAL MODEL'''
    
    ## Scaler transform our numeric variables
    feat_scale = scaler.feature_names_in_.tolist()
    df_scaled = pd.DataFrame(scaler.transform(pd.get_dummies(df)[feat_scale]),columns=feat_scale,index=df.index)
    
    ## make FAMD transformation
    famd_weight = df_famd.loc['weight']
    famd_mean = df_famd.loc['mean']
    
    # capture categorical columns by casting to dummy variables and performing FAMD weighting
    df_cat = pd.get_dummies(df)[df_famd.columns].copy()
    
    # FAMD transform
    df_cat = (df_cat/famd_weight)-famd_mean
    
    # concatenate our scaled dataframes
    df_scaled = pd.concat([df_scaled,df_cat],axis=1)
    
    # weight our scaled df before PCA
    df_scaled = df_scaled*s_wt
    
    # fill null values with mean before we cluster
    df_scaled = df_scaled.fillna(df_scaled.mean())
    
    # make sure our columns are ordered properly
    feat_pca = pca.feature_names_in_.tolist()
    df_scaled = df_scaled[feat_pca].copy()
    
    # PCA transform our dataframe
    df_ready = pd.DataFrame(pca.transform(df_scaled),index=df_scaled.index)
    
    return df_ready