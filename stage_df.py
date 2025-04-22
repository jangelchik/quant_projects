import pandas as pd
import numpy as np

def stage_df(df,
             fillna=False,
             null_outlier=False):
    
    '''
    PARAMETERS:
    df - Pandas DataFrame with null values
    fillna: bool - whether or not we want to fill null values
    
    RETURNS:
    df_clean - Pandas DataFrame with singular/irrelevant/string columns dropped and remaining categories in dummy state
    '''
    
    # firstly, we'll drop any all-null or singular columns
    s_nunique = df.nunique()
    s_null = df.count()
    
    lst_drop = df.columns[(s_nunique<2)|(s_null==0)].tolist()
    
    df.drop(columns = lst_drop,inplace=True)
    
    # keep track of the source columns for our categorical dummy variables
    s_type = df.dtypes.astype('str')
    lst_src = s_type[s_type.str.contains('object')].index.tolist()
    
    df_dum = pd.DataFrame()
    if len(lst_src)>0:
        df_dum = pd.get_dummies(df[lst_src],dtype='int')
    
    lst_dt = s_type[s_type.str.contains('datetime64')].index.tolist()
            
    df_fit = pd.concat([df.drop(columns=lst_src),df_dum],axis=1)
    
    if len(lst_dt)>0:
        for col in lst_dt:
            df_fit[col] = df_fit[col].values.astype('int')
        
     # fill nulls that may have been created
    if fillna == True:

        # determine if we should maintain nulls as outliers for tree models or if we should fill with means
        if null_outlier == True:
            for col in s_type[s_type.str.contains('float|int')].index:
                df_fit[col] = df_fit[col].fillna(-999)
                
        else:
            for col in s_type[s_type.str.contains('float|int')].index:
                df_fit[col] = df_fit[col].fillna(df_fit[col].mean())
                
    return df_fit,lst_src,lst_dt

