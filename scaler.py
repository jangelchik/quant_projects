import pandas as pd
import numpy as np

from stage_df import *

from typing import Union

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from yeo_johnson_sklearn import *

# Standardization Function
def scale_df(df,
             scale_type = 'std',
             famd=True,
             s_wt:Union[None,pd.Series]=None,
             yj=False):
    
    '''
    PARAMETERS:
    df - Pandas Dataframe: data to be scaled
    scale_type - str: ['std','mm'], whether to use mean & std dev or minmax scaling
    famd - boolean, whether or not we want to divide categorical variables by the sqrt of their probabilities for possible dimensionality reduction
    s_wt - None or Pandas Series, weights to apply to each feature/column in our feature set
    yj: bool - whether or not we want to perform a yeo_johnson power transform on the data
    
    RETURNS:
    *** In a dictionary ***
    df_scaled - Pandas DataFrame: scaled data
    scaler - Sklearn Scaler object: either minmax or standard values
    df_famd - Pandas DataFrame: the weights and mean values for transforming categorical features with FAMD
    s_wt - Pandas Series: the weights applied to our features
    '''
    
    # cast all variables to numeric
    df_scaled,lst_src,lst_dt = stage_df(df)
    
    # perform yeo-johnson tranform
    pt = PowerTransformer(standardize=False)
    if yj == True:
        df_scaled,pt = yeo_johnson_df(df_scaled)
    
    ## identify all categorical variables
    # must have 2 unique values, min val of 0, max val of 1
    df_cat = pd.DataFrame()
    lst_cat = df_scaled.columns[(df_scaled.nunique()==2)&(df_scaled.min()==0)&(df_scaled.max()==1)].tolist()
    if len(lst_cat)>0:
        df_cat = df_scaled[lst_cat].copy()
        
    ## identify continuous/ordinal variables and scale them
    # instantiate scaler
    scaler = StandardScaler()
    if scale_type != 'std':
        scaler = MinMaxScaler()
    # create df for scaled variables
    df_cont = pd.DataFrame(index=df_scaled.index)
    lst_cont = df_scaled.columns[~df_scaled.columns.isin(lst_cat)].tolist()
    # account for condition in which all variables are categorical
    if len(lst_cont)>0:
        df_cont=df_scaled[lst_cont].copy()

        # fit our scaler
        scaler.fit(df_cont)

        # tranform our continuous values
        df_cont = pd.DataFrame(scaler.transform(df_cont),columns=df_cont.columns,index=df_cont.index)
    
    # PERFORM FAMD ON CATEGORICAL DUMMY COLUMNS
    ## assume we do not need to transform data with FAMD
    df_famd = pd.DataFrame()
    if df_cat.columns.size > 0:
        df_famd = pd.DataFrame(np.full((df_cat.columns.size,2),1)).T
        df_famd.columns = df_cat.columns
        df_famd.index = ['weight','mean']
    
        # however, if we do want to use FAMD
        if famd == True:

            # capture probability
            count = df_cat.sum(numeric_only=True)
            p = count / df_cat.shape[0]

            # weight is sqrt of probability
            s_weight = p**(1/2)
            # divide column by squareroot of its probability
            df_cat = df_cat/s_weight

            # now mean center these columns
            s_mean = df_cat.mean(numeric_only=True)
            df_cat = df_cat - s_mean

            # capture our famd transformation
            df_famd = pd.concat([s_weight,s_mean],axis=1).rename(columns={0:'weight',1:'mean'}).T
        
        df_famd.index.name = 'params'
        
    # concatenate continuous and categorical variables back together
    df_scaled = pd.concat([df_cont,df_cat],axis=1)
                
    # multiply by feature weights, if relevant
    if type(s_wt).__name__== 'NoneType':
        s_wt = pd.Series(np.full(df_scaled.columns.size,1))
        s_wt.name = 'no_wt'
        s_wt.index = df_scaled.columns
        
    df_scaled = (df_scaled * s_wt)
    # return the fit df, the scaling df, the famd df, and the weighting series
    return {'df_scaled':df_scaled,
            'pt':pt,
            'scaler':scaler,
            'df_famd':df_famd,
            's_wt':s_wt}


def transform_scale(df:pd.DataFrame,
                    scaler:Union[StandardScaler,MinMaxScaler],
                    df_famd:pd.DataFrame,
                    s_wt:pd.Series):
    
    '''
    PARAMETERS:
    df: Pandas DataFrame - our feature set
    scaler: [StandardScaler,MinMaxScaler] - a fitted sklearn scaler object
    df_famd: Pandas DataFrame - our weights and means for famd transformation
    s_wt: Pandas Series - our feature weights, if applicable
    
    RETURNS:
    df_scaled: Pandas DataFrame - our feature set scaled according to an already fit scaling process 
    '''
    
    ## stage our df
    df_staged,lst_src,lst_dt = stage_df(df)
    
    ## Scaler transform our numeric variables
    feat_scale = scaler.feature_names_in_.tolist()
    df_cont = pd.DataFrame(scaler.transform(df_staged[feat_scale]),columns=feat_scale,index=df.index)
    
    ## make FAMD transformation
    df_cat = pd.DataFrame()
    lst_cat = df_staged.columns[(df_staged.nunique()==2)&(df_staged.min()==0)&(df_staged.max()==1)].tolist()
    if len(lst_cat)>0:
        df_cat = pd.DataFrame(df_staged[lst_cat],columns=lst_cat,index=df.index)
        
    if df_famd.columns.size>0:
        famd_weight = df_famd.loc['weight']
        famd_mean = df_famd.loc['mean']

        # FAMD transform
        df_cat = (df_cat/famd_weight)-famd_mean
    
    # concatenate our scaled dataframes
    df_scaled = pd.concat([df_cont,df_cat],axis=1)
    
    # weight our scaled df before PCA
    df_scaled = df_scaled*s_wt
    
    return df_scaled