# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 20:45:55 2022

@author: Eier
"""

# finds the locations where horizons intersect the cross-section


import segyio
import pandas as pd
import numpy as np
#fault seis.segy
#wedge seis.segy'
# file_seis = 'data\case Oseberg\case oseberg LFM imp.segy'
# hor_file = 'data\horizons\Drake 100ms below'
# hor_file = 'data\horizons\Drake LFM'
# hor_file = 'data\horizons\Brent LFM'
# hor_file = 'data\horizons\Shetland LFM'
# hor_file = 'data\horizons\Shetland 50 ms above'

# cross section
file_imp = 'data\case F3\Seis Inv depth Random line [2D Converted].segy'

f = segyio.open(file_imp, ignore_geometry=True)
#'data\case F3\F3-Horizon-FS8 (Z)', 

# horizons
hor = ['data\case F3\F3-Horizon-MFS4 (Z)', 'data\case F3\F3-Horizon-Truncation (Z)']

# for each horizons
for hor_file in hor:

    # get TWT of section
    sec = segyio.tools.metadata(f)
    TWT = -sec.samples
    top_twt = TWT.max()
    base_twt = TWT.min()
    l = len(f.header)
    
    s = len(f.trace[1])
    
    # inter = (top_twt-base_twt)/161
    
    inter = (top_twt-base_twt)/s
    
    resx = np.zeros((l, s))
    resy = np.zeros((l, s))
    
    
    for trace in range(l):
        scale = f.header[trace][segyio.TraceField.SourceGroupScalar]
        # get the coordinates of the cross-section
        if scale <0:
        
            x = f.header[trace][segyio.TraceField.SourceX]/abs(scale)
            y = f.header[trace][segyio.TraceField.SourceY]/abs(scale)
        
        else:
            x = f.header[trace][segyio.TraceField.SourceX]*abs(scale)
            y = f.header[trace][segyio.TraceField.SourceY]*abs(scale)
        
        resx[trace] = x
        resy[trace] = y
        
        samples = f.trace[trace]
            
        
        
    #############################
    
    TWT_array = np.zeros(len(resx))
    
    # load horizon
    hor = pd.read_csv(hor_file, header = None, sep = ' ')
    z = np.zeros(len(hor))
    
    for coo in range(len(resx)):
        print(coo)
        for i in range(len(hor)):
            # horizon point
            point_a = np.array([hor[0][i], hor[1][i]])
            
            # cross-section point
            point_b = np.array([resx[coo, 0], resy[coo, 0]])
            
            # distance between points
            dist = np.sqrt((point_a[0]-point_b[0])*(point_a[0]-point_b[0])+
                           (point_a[1]-point_b[1])*(point_a[1]-point_b[1]))
            
            z[i] = dist
            
        # get smallest distance
        in_min = np.argmin(z)
        
        TWT = hor.iloc[in_min][2]
        TWT_array[coo] = -TWT
        
    np.save(hor_file+'TWT',TWT_array)
f.close()
    
# np.load(hor_file+'TWT'+'.npz')