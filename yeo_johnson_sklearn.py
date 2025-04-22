import pandas as pd
import numpy as np

from scipy.stats import boxcox,yeojohnson

from typing import Union

from stage_df import *

import multiprocessing as mp

from sklearn.preprocessing import PowerTransformer

def yeo_johnson_df(df:pd.DataFrame):
    
    '''
    PARAMETERS:
    df: Pandas DataFrame
    
    RETURNS:
    df_bc: Pandas DataFrame - box-cox transformed columns
    d_bc: column names and their corresponding lambda parameters
    '''
    
    df_fit,lst_src,lst_dt = stage_df(df)
    
    df_cat = df[df_fit.columns[(df_fit.nunique()==2)&(df_fit.min()==0)&(df_fit.max()==1)]].copy()
    df_cont = df_fit[df_fit.columns[~df_fit.columns.isin(df_cat.columns)]].copy()
    
    pt = PowerTransformer(standardize=False)
    pt.fit(df_cont)
    
    df_cont = pd.DataFrame(pt.transform(df_cont),columns=df_cont.columns,index=df_cont.index)
    
    df_yj = pd.concat([df_cont,df_cat],axis=1)

    return df_yj,pt