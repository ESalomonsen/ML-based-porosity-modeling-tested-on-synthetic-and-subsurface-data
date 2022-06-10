# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:26:19 2021

@author: Eier
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class load_well:
    """
    Handles the extraction and editing of the well-data.
    This can be real well data or synthetic well data from a 2D array of synthetic data.
    """
    def __init__(self, file_name):
        """
        Handles the extraction and editing of the well-data.
        This can be real well data or synthetic well data from a 2D array of synthetic data.

        Parameters
        ----------
        file_name : str
            The file location of the well data.

        Returns
        -------
        None.

        """
        
        self.file = file_name
        
    def from_excel(self):
        """
        Imports well-log data from an excel file.

        Returns
        -------
        data : pandas.DataFrame
            dataframe of the well-logs. 
            one column per well-log.

        """
        
        data= pd.read_excel(self.file, header = 2) #file_name = 'data_2d_wedge_F3\F03_2_por_eff.xlsx'
        
        data.drop('Unnamed: 0', inplace = True, axis = 1)
        data = data.dropna()
        #data = data.set_index('MD')
        data.drop('MD', inplace = True, axis = 1)
        
        return data
    def from_csv(self, drop_MD = True):
        """
        Imports well-log data from an csv file.

        Parameters
        ----------
        drop_MD : bool, optional
            if True; will drop the 'MD' column. The default is True.

        Returns
        -------
        data : pandas.DataFrame
            dataframe of the well-logs. 
            one column per well-log.
        MD : numpy array
            the measured depth.

        """
        
        data = pd.read_csv(self.file)
        MD = data['MD'].to_numpy()
        if drop_MD==True:
            
            data.drop('MD', inplace = True, axis = 1)
            
        return data, MD
        
    
    def from_synthetic(self, grid_list, name_list, col = 150):
        """
        Imports a trace from one or more synthetic sections as if it was a well-log.

        Parameters
        ----------
        grid_list : list
            list of 2d arrays representing the cross-sections.
        name_list : list
            list of grid names, for example ['imp', 'por'].
        col : int, optional
            trace number. The default is 150.

        Returns
        -------
        df : pandas.DataFrame
            dataframe of the well-logs. 
            one column per well-log.

        """
        n = 0
        l = len(grid_list[0].T)
        data_ar = np.zeros((l, len(grid_list)))
        
        # get trace = col in every cross-section
        for grid in grid_list:
            data_ar[:,n] = grid[col] # row in grid should be verticle ---> .T provides correct plot
            n = n+1
            
        # compile traces into dataframe
        df = pd.DataFrame(data_ar, columns = name_list)
        return df


# def win_select(data, data_name = 'Por.Eff.',win =4):
    
#     data_col= data[data_name].to_numpy()
    
#     win_2 = int(win/2)
#     l = len(data_col)
    
#     for w in range(win_2):
#         w = w+1
        

#         # start with 1 and -1
#         #
#         up = np.zeros(l)
#         up[:] = np.nan
#         up[w:l] = data_col[0:l-w]
        
#         down = np.zeros(l)
#         down[:] = np.nan
#         down[0:l-w] = data_col[0+w:l]
        
#         data['+{} wind'.format(w)] = up
#         data['-{} wind'.format(w)] = down
#     return data
    

# df = pd.read_excel('data\case F3\F02_1.xlsx')
# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# df_por = pd.DataFrame()
# df_por['MD'] = df['MD']
# df_por['Por.Eff.'] = df['Por.Eff.']
# df_por = df_por.dropna()

# df_imp = pd.DataFrame()
# df_imp['MD'] = df['MD.1']
# df_imp['P-imp.'] = df['P-imp.']
# df_imp = df_imp.dropna()

# por_MD_lim = [df_por['MD'].min(), df_por['MD'].max()]

# df_imp = df_imp[df_imp['MD']>por_MD_lim[0]]
# df_imp = df_imp[df_imp['MD']<por_MD_lim[1]]


# df_imp = df_imp.reset_index()
# df_por = df_por.reset_index()

# df_well = pd.DataFrame()

# def nearest(number, arr):
#     """
#     Gets index of number nearest the "number"

#     Parameters
#     ----------
#     number : TYPE
#         DESCRIPTION.
#     arr : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     TYPE
#         DESCRIPTION.

#     """
#     search = abs(arr-number)
#     m = search.min()
#     return np.where(search == m)[0]

# md_list = []
# imp_list = []
# por_list = []


# for n, i in enumerate(df_imp['MD']):
    
#     arr = df_por['MD'].to_numpy()
    
#     ind = nearest(i, arr)[0]
    
#     md = df_imp.iloc[n]['MD']
#     md_list.append(md)
#     imp = df_imp.iloc[n]['P-imp.']
#     imp_list.append(imp)
#     por = df_por.iloc[ind]['Por.Eff.']
#     por_list.append(por)
    
# df_well['MD'] = md_list
# df_well['imp'] = imp_list
# df_well['por'] = por_list
    
# plt.scatter(df_well['imp'], df_well['por'])





    
    
    
    
    
    
    
    