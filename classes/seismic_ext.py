# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:54:10 2021

@author: Eier
"""


import segyio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi



class seismic_ext:
    """
    Exctracts sections using segyio
    """
    
    def __init__(self, file_name):
        """
        initialise class
        
        Parameters
        ----------
        file_name : float
            file path of the segy file.

        Returns
        -------
        None.

        """
        
        self.file = file_name
        
    def syn_seismic(self, plot = False, scaling = False):
        """
        uses segyio to get the section as a numpy grid.
        also gets the depth ready for the class plot results.

        Parameters
        ----------
        plot : bool, optional
            Should the sections be plotted for evaluation? The default is False.
        scaling : bool, optional
            should the section numpy grid be standardized? The default is False.

        Returns
        -------
        grid : 2D numpy array
            section as numpy array, traces as rows, depth as columns.
        extent : list
            axis extent as traces and TWT, for later plotting.

        """
        
        # open file
        f = segyio.open(self.file, ignore_geometry=True)
        
        # get the metadata, incl. TWT and trace ranges
        sec = segyio.tools.metadata(f)
        TWT = -sec.samples # get depth data
        TWT_max = TWT.max()
        TWT_min = TWT.min()
        trace_min = f.header[0][segyio.TraceField.TraceNumber]
        trace_max= f.header[-1][segyio.TraceField.TraceNumber]
        
        extent = [trace_min, trace_max, TWT_min, TWT_max] # get depth and trace numbers for the plots
        
        # get cross-section
        grid = f.trace.raw[:]
        f.close()
        
        if plot == True:
            vm = np.percentile(grid, 99.5)
            plt.imshow(grid.T, cmap='jet', aspect='auto', vmin=-vm, vmax=vm)
            plt.title(self.file)
            plt.show()
            
        else:
            pass
        
        if scaling == True:
            grid = (grid - np.mean(grid)) / np.std(grid) # standardize data
        
        self.grid = grid
        
        return grid, extent
