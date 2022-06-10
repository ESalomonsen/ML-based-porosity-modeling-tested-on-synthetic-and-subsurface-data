# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:42:06 2022

@author: Eier
"""

# the logs of a well may not match
# this script matches the logs 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

well_files = ['data\case F3\F02_1.xlsx', 'data\case F3\F03_2.xlsx']

new_well_files = ['data\case F3\F02_1_new.txt', 'data\case F3\F03_2_new.txt']

for k,file in enumerate(well_files):
    df = pd.read_excel(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df_por = pd.DataFrame()
    df_por['MD'] = df['MD']
    df_por['Por.Eff.'] = df['Por.Eff.']
    df_por = df_por.dropna()
    
    df_imp = pd.DataFrame()
    df_imp['MD'] = df['MD.1']
    df_imp['P-imp.'] = df['P-imp.']
    df_imp = df_imp.dropna()
    
    por_MD_lim = [df_por['MD'].min(), df_por['MD'].max()]
    
    df_imp = df_imp[df_imp['MD']>por_MD_lim[0]]
    df_imp = df_imp[df_imp['MD']<por_MD_lim[1]]
    
    
    df_imp = df_imp.reset_index()
    df_por = df_por.reset_index()
    
    df_well = pd.DataFrame()
    
    def nearest(number, arr):
        """
        Gets index of number nearest the "number"
    
        Parameters
        ----------
        number : TYPE
            DESCRIPTION.
        arr : TYPE
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        search = abs(arr-number)
        m = search.min()
        return np.where(search == m)[0]
    
    md_list = []
    imp_list = []
    por_list = []
    
    
    for n, i in enumerate(df_imp['MD']):
        
        arr = df_por['MD'].to_numpy()
        
        ind = nearest(i, arr)[0]
        
        md = df_imp.iloc[n]['MD']
        md_list.append(md)
        imp = df_imp.iloc[n]['P-imp.']
        imp_list.append(imp)
        por = df_por.iloc[ind]['Por.Eff.']
        por_list.append(por)
        
    df_well['MD'] = md_list
    df_well['imp'] = imp_list
    df_well['por'] = por_list
        
    plt.scatter(df_well['imp'], df_well['por'])
    
    df_well.to_csv(new_well_files[k], index = False)