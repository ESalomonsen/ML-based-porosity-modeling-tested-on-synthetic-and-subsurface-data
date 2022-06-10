# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:18:12 2022

@author: Eier
"""

# upscales the well logs based on an existing already upscaled well log

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def nearest(number, arr):
    """
    Gets index of number nearest the "number"

    Parameters
    ----------
    number : float
        the value to find in arr.
    arr : numpy array
        array of values.

    Returns
    -------
    int
        the index where arr is closest to number.

    """
    search = abs(arr-number)
    m = search.min()
    return np.where(search == m)[0]

# get upscaled well log
upscale = 'data\case F3\AI resampled.xlsx'

up_df = pd.read_excel(upscale)
up_df = up_df.loc[:, ~up_df.columns.str.contains('^Unnamed')]

# well logs to be upscaled
well_files = ['data\case F3\F02_1_new', 'data\case F3\F03_2_new']

# for each well
for k,file in enumerate(well_files):
    df = pd.read_csv(file+'.txt')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df_por = pd.DataFrame()
    df_por['MD'] = df['MD']
    df_por['por'] = df['por']
    df_por = df_por.dropna()
    
    df_imp = pd.DataFrame()
    df_imp['MD'] = df['MD']
    df_imp['imp'] = df['imp']
    df_imp = df_imp.dropna()
    
    plt.scatter(df['imp'], df['por'])
    plt.show()
    
    por_MD_lim = [df_por['MD'].min(), df_por['MD'].max()]
    
    df_imp = df_imp[df_imp['MD']>por_MD_lim[0]]
    df_imp = df_imp[df_imp['MD']<por_MD_lim[1]]
    
    
    df_imp = df_imp.reset_index()
    df_por = df_por.reset_index()
    
    df_well = pd.DataFrame()
    
    md_list = []
    imp_list = []
    por_list = []
    
    # for every point in the upscaled well-log MD
    for n, i in enumerate(up_df['MD']):
        
        arr_por = df_por['MD'].to_numpy()
        arr_imp = df_imp['MD'].to_numpy()
        
        # find the closest point in the real well-logs and save
        ind_por = nearest(i, arr_por)[0]
        ind_imp = nearest(i, arr_imp)[0]
        
        md = up_df.iloc[n]['MD']
        md_list.append(md)
        
        imp = df_imp.iloc[ind_imp]['imp']
        imp_list.append(imp)
        
        por = df_por.iloc[ind_por]['por']
        por_list.append(por)
    
    # store the new upscaled well logs in a dataframe
    df_well['MD'] = md_list
    df_well['imp'] = imp_list
    df_well['por'] = por_list
        
    plt.scatter(df_well['imp'], df_well['por'])
    plt.show()
    # df_well.to_csv(file+'_upsacale.txt', index = False)