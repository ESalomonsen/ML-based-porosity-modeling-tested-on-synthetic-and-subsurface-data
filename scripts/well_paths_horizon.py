# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:44:24 2022

@author: Eier
"""
# implements the horizon locations to the well.
# the well path is given with coordinates (interpolation is requiered)
# the horizon point set is given with coordinates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def TWT_from_MD(MD):
    z = np.load('TDR\_fit TDR result.npy')
    p = np.poly1d(z)
    
    return p(MD)


file = 'data\case F3\F03_2 well path'
# file = 'data\case Oseberg\w30_9_J_13_well_path'
hor_file1 = 'data\case F3\F3-Horizon-FS8 (Z)'
hor_file2 = 'data\case F3\F3-Horizon-MFS4 (Z)'
hor_file3 = 'data\case F3\F3-Horizon-Truncation (Z)'

hor_file_list = [hor_file1, hor_file2, hor_file3]

##################################################################################################
df = pd.read_csv(file, skiprows=17, sep = ' ')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna()
############## from https://stackoverflow.com/questions/66466080/python-pandas-insert-empty-rows-after-each-row #####################
n = 10
new_index = pd.RangeIndex(len(df)*(n+1))
new_df = pd.DataFrame(np.nan, index=new_index, columns=df.columns)
ids = np.arange(len(df))*(n+1)
new_df.loc[ids] = df.values
df = new_df
################################################# interpolate #########################################################################################################
df = df.interpolate()
df = df.reset_index()
#################################### MD to Z in well path ##############################################################
z = np.polyfit(df['MD'], df['Z'], 12)
np.save(file[:18]+'MD to Z '+file[18:], z)
##################################################################################################

hor_location_5S = pd.DataFrame()

# for each horizon
for hor_file in hor_file_list:
    print(hor_file)

    ##########################################################################################################################################################
    hor = pd.read_csv(hor_file, header = None, sep = ' ')
    hor[2] = -hor[2]
    
    ################################################### reduce the search area for well location #######################################################################################################
    upperX = df['X'].max()+10
    upperY = df['Y'].max()+10
    
    lowerX = df['X'].min()-10
    lowerY = df['Y'].min()-10
    
    hor = hor.drop(hor[(hor[0] < lowerX)].index)
    hor = hor.drop(hor[(hor[0] > upperX)].index)
    
    hor = hor.drop(hor[(hor[1] < lowerY)].index)
    hor = hor.drop(hor[(hor[1] > upperY)].index)
    hor = hor.reset_index()
    ##########################################################################################################################################################
    # df['TWT'] = TWT_from_MD(df['MD'])
    
    
    # reserve memory
    z = np.zeros(len(hor)*len(df))
    twt = np.zeros(len(hor)*len(df))
    n = 0
    
    # follow the well path interpolated points
    for coo in range(len(df)):
        print(coo)
        for i in range(len(hor)):
            # horizon point
            point_a = np.array([hor[0][i], hor[1][i], hor[2][i]])
            
            # well point
            point_b = np.array([df['X'][coo], df['Y'][coo], df['Z'][coo]])
            
            # distance between points
            dist = np.sqrt((point_a[0]-point_b[0])*(point_a[0]-point_b[0])+
                            (point_a[1]-point_b[1])*(point_a[1]-point_b[1])+
                            (point_a[2]-point_b[2])*(point_a[2]-point_b[2]))
            z[n] = dist
            twt[n] = hor[2][i]
            n+=1
    
    # find the two closest points
    in_min = np.argmin(z)
    
    # record the horizon location in the well path
    TWT_loc = twt[in_min]
    hor_location_5S[hor_file[14:]] = [TWT_loc]
    
# compile results
hor_location_5S.to_csv(file+' horizon loc'+'.txt', index = False, encoding = 'utf-8')

# plt.plot(df['X'], df['Y'])
# plt.show()
# plt.plot(df['X'], df['Z'])
# plt.show()
# plt.plot(df['Y'], df['Z'])
# plt.show()

    
    