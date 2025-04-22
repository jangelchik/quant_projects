import pandas as pd
import numpy as np

from scaler import *

from find_elbow import *

import warnings

from typing import Union

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler,MinMaxScaler
    
def pca(df_scaled:pd.DataFrame,
        pt:PowerTransformer,
        scaler:Union[StandardScaler,MinMaxScaler],
        df_famd:pd.DataFrame,
        s_wt:pd.Series):
    
    '''
    PARAMETERS:
    df_scaled: Pandas DataFrame - scaled feature set
    scaler: Sklearn scaler object - Either Standard (z-score) or MinMax
    df_famd: Pandas DataFrame - the weights and mean values for transforming categorical features with FAMD
    s_wt: Pandas Series - the series used to weight the feature set
    memoryless: bool - whether or not we want to return the fit DataFrame, or just the transformation objects
    
    RETURNS:
    *** In a dictionary ***
    df_pca - Pandas DataFrame: PCA-Transformed feature set
    pca_ - Sklearn PCA object: fit PCA object
    evr - float: Explained Variance Ratio of our PCA-transformed feature set
    df_scaled - Pandas DataFrame: scaled feature set
    scaler: Sklearn scaler object - Either Standard (z-score) or MinMax
    df_famd - Pandas DataFrame: the weights and mean values for transforming categorical features with FAMD
    s_wt - Pandas Series: series used to weight the feature set
    '''
    
    # Only fit with unique rows
    X = df_scaled.drop_duplicates()
    
    ## Use PCA to reduce dimensions while capturing a threshold of variability
    pca = PCA()
    pca.fit(X)
    
    vrs = pca.explained_variance_ratio_
    
    # capture the explained variance ratios for each dimension
    cum_sums = np.cumsum(vrs)
    elbows = find_elbow(cum_sums,'concave')

    
    # default to 95% explained variance, unless we found optimal number of dimensions with elbow/knee
    pca = PCA(0.95)
    
    if len(elbows) > 0:
        # take maximum elbow value
        pca = PCA(np.max(elbows))
        
    # finally, fit PCA object with optimal number of dimensions and transform our data
    pca.fit(X)
    
    df_pca = pd.DataFrame(pca.transform(df_scaled))
    df_pca.index = df_scaled.index
  
    return {'df_pca':df_pca,
            'pca':pca,
            'df_scaled':df_scaled,
            'pt':pt,
            'scaler':scaler,
            'df_famd':df_famd,
            's_wt':s_wt}

def transform_pca(df:pd.DataFrame,
                  scaler:Union[StandardScaler,MinMaxScaler],
                  df_famd:pd.DataFrame,
                  s_wt:pd.Series,
                  pca:PCA):
    
    '''
    PARAMETERS:
    df: Pandas DataFrame - our feature set
    scaler: [StandardScaler,MinMaxScaler] - a fitted sklearn scaler object
    df_famd: Pandas DataFrame - our weights and means for famd transformation
    s_wt: Pandas Series - our feature weights, if applicable
    pca: PCA - a fitted sklearn PCA object
    
    RETURNS:
    df_scaled: Pandas DataFrame - our feature set scaled according to an already fit scaling process 
    '''
    
    # first, scale our dataset
    df_scaled = transform_scale(df,scaler,df_famd,s_wt)
    
    # now transform with PCA
    df_pca = pd.DataFrame(pca.transform(df_scaled.fillna(0)))
    df_pca.index = df_scaled.index
    
    return df_pca
    
    
    
    
