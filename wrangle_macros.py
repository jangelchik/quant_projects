import pandas as pd
import numpy as np
import requests
import datetime as dt

import os
os.system('pip install --upgrade kneed')
os.system('pip install --upgrade fredapi')
os.system('pip install --upgrade hdbscan')

from scaler import *
from pca import *

from fredapi import Fred

import hdbscan
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

def wrangle_macros(mmr_path:str,
                   date_thresh=None,
                   date_fit_thresh=None):

    '''
    PARAMETERS:
    mmr_path:str - file path to latest MMR index numbers from Manheim website (locally saved file)

    RETURNS:
    df_macro:pd.DataFrame - Dataframe containing our wrangled and PCA processed data
    '''

    ### read in Manheim
    df_mmr = pd.read_excel(mmr_path,engine='openpyxl')
    
    df_mmr.rename(columns={'Unnamed: 0':'date',
                   'Manheim Index $ amount NSA':'mmr_index'},inplace=True)
    
    df_mmr.set_index('date',inplace=True)
    
    df_mmr.index.name=None
    
    s_mmr = df_mmr['mmr_index'].copy()
    
    s_mmr.index = pd.to_datetime(s_mmr.index.strftime('%Y-%m'))
    
    ### Import Macro Data From FRED API
    key = 'd660ceb498015ebea6dd84e8a932e8fd'
    
    fred = Fred(api_key=key)
    
    d_data = dict()
    
    for data in '''IPN213111N - Drilling Oil and Gas Wells
    IPG211N - Oil and Gas Extraction
    POILWTIUSDM - WTI Crude
    M2NS - M2 Supply
    CEU0500000001 - PrivateEmp
    CEU0500000003 - Avg Hourly Earnings, PrivateEmp
    AISRSA - InvSalesRatio
    TOTALNSA - Total Vehicle Sales
    HTRUCKSNSA - Retail Sales Heavy Weight Trucks
    LTRUCKNSA - Retail Sales Light Weight Trucks
    DLTRUCKSNSA - Retail Sales Domestic Light Weight Trucks
    FLTRUCKSNSA - Retail Sales Foreign Light Weight Trucks
    TTLCON - Total Construction Spend
    POPTHM - US Population
    EXPTOTUS - US Exports
    IMPTOTUS - US Imports
    IPG2122N - Metal Ore Mining
    PURANUSDM - Uranium
    PCOPPUSDM - Copper
    PZINCUSDM - Zinc
    PTINUSDM - Tin
    PLEADUSDM - Lead
    PALUMUSDM - Aluminum
    PNICKUSDM - Nickel
    PIORECRUSDM - Iron Ore
    PWHEAMTUSDM - Wheat
    PSUNOUSDM - Sunflower Oil
    PBARLUSDM - Barley
    PMAIZMTUSDM - Corn
    PPOILUSDM - Palm Oil
    PCOTTINDUSDM - Cotton
    PRUBBUSDM - Rubber
    PBANSOPUSDM - Bananas
    PCOCOUSDM - Cocoa
    PSOYBUSDM - Soy
    PSUGAISAUSDM - Sugar
    PORANGUSDM - Orange
    PROILUSDM - Rapeseed Oil
    PBEEFUSDM - Beef
    PPOULTUSDM - Poultry
    PPORKUSDM - Swine
    PLAMBUSDM - Lamb
    PLOGOREUSDM - Softlogs'''.split('\n'):
    
        d_data[data.split(' - ')[0]] = data.split(' - ')[1]
    
    ## iterate over series, pulling data from FRED
    lst_s = [s_mmr]
    
    for k,v in d_data.items():
    
        s = fred.get_series(k.replace(' ',''))
        s.name = v
    
        s.index = pd.to_datetime(s.index.strftime('%Y-%m'))
    
        lst_s.append(s)
    
    
    
    #### Consider fossil fuels vs alternatives instead of "Energy"
    
    d_groups = {'Vehicle Sales':['mmr_index',
                                 'InvSalesRatio',
                                 'Total Vehicle Sales',
                                 'Retail Sales Heavy Weight Trucks',
                                 'Retail Sales Light Weight Trucks',
                                 'Retail Sales Domestic Light Weight Trucks',
                                 'Retail Sales Foreign Light Weight Trucks'],
                'Energy':['Drilling Oil and Gas Wells',
                          'Oil and Gas Extraction',
                          'WTI Crude',
                          'Uranium',
                          'Corn'],
                'M2 Supply':['M2 Supply'],
                'Employment':['PrivateEmp',
                              'Avg Hourly Earnings, PrivateEmp'],
                'Population':['US Population'],
                'Construction':['Total Construction Spend'],
                'US ImpExp':['US Exports',
                             'US Imports'],
                'Metals':['Metal Ore Mining',
                          'Copper',
                          'Zinc',
                          'Tin',
                          'Lead',
                          'Aluminum',
                          'Nickel',
                          'Iron Ore'],
                'Crops':['Wheat',
                         'Sunflower Oil',
                         'Barley',
                         'Corn',
                         'Palm Oil',
                         'Cotton',
                         'Rubber',
                         'Bananas',
                         'Cocoa',
                         'Soy',
                         'Sugar',
                         'Orange',
                         'Rapeseed Oil'],
                'Livestock':['Beef',
                             'Poultry',
                             'Swine',
                             'Lamb'],
                'Lumber':['Softlogs']
               }
    
    '|'.join(list(d_groups.keys()))
    
    ## create df object to join all series together
    df_macro = pd.concat(lst_s,axis=1).sort_index().asfreq('MS')

    if date_thresh:
        df_macro = df_macro[df_macro.index<=date_thresh].copy()
    
    # take yearly rolling mean to account for seasonality --- ### test purely YoY chg for now
    # df_roll = df_macro.rolling(window=12).mean()
    df_roll = df_macro.copy()
    
    ## Now, Perform PCA on our Macro Variables
    
    # now get YoY pct changes in the rolling mean
    df_roll_yoy = (df_roll  - df_roll.shift(12))/df_roll.shift(12)
    
    df_roll_yoy = df_roll_yoy.join(pd.date_range(df_roll_yoy.index[-1],dt.datetime.now().date(),freq='MS').to_frame(),how='outer').drop(columns=[0])
    
    df_roll_yoy.tail()
    
    ## now get monthly lagged YoY values for prior 12 months
    lst_df = [df_roll_yoy]
    for i in range(1,13):
    
        df_ = df_roll_yoy.shift(i)
    
        df_.columns = df_.columns + f'_lag{i}'
    
        lst_df.append(df_)
    
    # forward fill and drop null rows
    df_lagged = pd.concat(lst_df,axis=1).ffill().dropna()  
    
    ### perform PCA on our macro variables
    
    lst_df = []
    
    for k,v in d_groups.items():
    
        lst_cols = v
    
        df_ = df_lagged[df_lagged.columns[df_lagged.columns.str.contains('|'.join(lst_cols))]].copy()

        df_all = df_.copy()

        if date_fit_thresh:
            df_ = df_[df_.index<=date_fit_thresh].copy()
    
        ## scale our macro variables for PCA
        d_scale = scale_df(df_,
                           scale_type = 'mm',
                           famd=False,
                           s_wt=None,
                           yj=True)
        
        ## perform PCA
        d_pca = pca(d_scale['df_scaled'], 
                    d_scale['pt'], d_scale['scaler'],
                    d_scale['df_famd'],
                    d_scale['s_wt'])
        
        # now yj transform all time within date thresh based on results from our fit thresh
        df_macro = pd.DataFrame(d_scale['pt'].transform(df_all),columns=df_all.columns,index=df_all.index)
        ## now scale all variables
        df_macro = pd.DataFrame(d_scale['scaler'].transform(df_macro),columns=df_macro.columns,index=df_macro.index)
        ## now perform pca on all variables
        df_macro = pd.DataFrame(d_pca['pca'].transform(df_macro),index=df_macro.index)
    
        print(k,df_macro.shape[1])
        
        df_macro.columns = f'{k}_macro_dim_' + (df_macro.columns+1).astype('str')
    
        lst_df.append(df_macro)
    
    ## concatenate PCA results back together
    df_macro = pd.concat(lst_df,axis=1)
    
    df_macro.index.name='date'

    ## NOW CLUSTER OUR MACRO DATA INTO EPOCHS
    X = df_macro.drop_duplicates()
    
    ## min max scale
    s_min = X.min()
    s_max = X.max()
    
    X = (X-s_min) / (s_max-s_min)
    
    lst_db_score = []
    lst_adj_score = []
    lst_tup_param = []
    
    for min_size in [int(np.log(X.shape[0])),int(np.sqrt(X.shape[0]))]:
    
        ### select for epsilon values based on min pts
        # instantiate nearest neighbors model and get distances to optimize for epsilon
        neigh = NearestNeighbors(n_neighbors= min_size , metric = 'euclidean')
    
        neigh.fit(X)
    
        distances, indices = neigh.kneighbors(X)
    
        # isolate the unique distances of the nth nearest neighbor
        distances = np.sort(distances[:,-1])
    
        # look for the inflection point
        lst_eps = find_elbow(distances,cv='convex')
    
        for eps in lst_eps:
    
            for min_samp in [int(np.log(X.shape[0])),int(np.sqrt(X.shape[0]))]:
    
                for method in 'eom leaf'.split():
                    ## fit HDBSCAN model to identify outliers
                    try:
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,
                                                    min_samples=min_samp,
                                                    cluster_selection_epsilon=eps,
                                                    cluster_selection_method = method)
                        clusterer.fit(X)
                
                        lst_tup_param.append((min_size,min_samp,eps,method))
                
                        lst_db_score.append(davies_bouldin_score(X, clusterer.labels_))
                        lst_adj_score.append(davies_bouldin_score(X, clusterer.labels_)*(clusterer.labels_[clusterer.labels_==-1].size/clusterer.labels_.size)*(np.unique(clusterer.labels_[clusterer.labels_!=-1]).size))
                    except:
                        pass
    
    
    params = lst_tup_param[np.argmin(lst_adj_score)]
    
    min_size = params[0]
    min_samp = params[1]
    eps = params[2]
    method = params[3]
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,
                                min_samples=min_samp,
                                cluster_selection_epsilon=eps,
                                cluster_selection_method = method)
    clusterer.fit(X)
    
    # reverse transform 
    X[s_max.index] = X[s_max.index]*(s_max - s_min) + s_min
    
    X['Cluster'] = clusterer.labels_
    
    X['Cluster'] = X['Cluster']+1
    
    idx_out = X[X['Cluster'] == 0].index
    
    X.loc[idx_out,'Cluster'] = -1
    
    df_macro.loc[X.index,'Cluster'] = X['Cluster'].copy().astype('str')
    
    # isolate our latest macro values to build a current macro outlook
    df_last = df_macro.iloc[-1]
    df_last.loc['Cluster'] = f"Current, {dt.datetime.now().date().strftime('%Y-%m')}"
    
    df_last = df_last.to_frame().T.set_index('Cluster')
    
    # build our prediction space clusters
    df_cluster = pd.concat([df_macro[df_macro['Cluster']!=-1].groupby('Cluster').median(),df_last],axis=0)
    df_cluster.index.name = 'Cluster'
    
    df_c = df_macro[['Cluster']].copy()
    
    df_c.to_csv('macro_grouped_cluster_dates.csv')
    
    return df_macro,df_cluster
    
