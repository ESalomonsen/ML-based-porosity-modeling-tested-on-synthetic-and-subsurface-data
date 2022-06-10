# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:22:11 2022

@author: Eier
"""
import matplotlib.pyplot as plt
import numpy as np
from classes.usefull_functions import usefull_functions
class plot_results:
    
    def __init__(self):
        self.sel = 'placeholder'
        
    def plot_generic(grid, vmin, vmax, title, xlab, ylab, extent,cmap = 'jet',
                     vcut=False, hcut = False):
        """
        Plot one cross-section

        Parameters
        ----------
        grid : numpy array
            2D array of the cross-section.
        vmin : float
            minimum value of the plotted array.
        vmax : float
            maximum value of the plotted array.
        title : str
            title of the plot.
        xlab : str
            x label.
        ylab : str
            y label.
        extent : list
            axis extent as traces and TWT.
        cmap : str, optional
            colormap. The default is 'jet'.
        vcut : list, optional
            vertical slice of the array (TWT). The default is False.
        hcut : list, optional
            horizontal slice of the array (traces). The default is False.

        Returns
        -------
        None.

        """
        
        if vcut == False:
            pass
        else:
            TWT = np.linspace(extent[3], extent[2], len(grid[1, :]))
            extent[3] = TWT[vcut[0]:vcut[1]][0]
            extent[2] = TWT[vcut[0]:vcut[1]][-1]
            
            grid = grid[:, vcut[0]:vcut[1]]

        if hcut == False:
            pass
        else:
            grid = grid[hcut[0]:hcut[1],:]


        plt.imshow(grid.T, cmap=cmap, aspect='auto',vmin=vmin, vmax=vmax, extent=extent) 
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.colorbar()
        plt.show()
        
    def grids_comp(self, grid1, grid2, vcut=False, hcut = False, cmap = 'jet', 
                   titles = ['True','Estimate', 'MSE'], file_name = False,
                   extent=[-1,1,-1,1], ylab = 'TWT', vmin =0.2, vmax = 0.4):
        """
        plot two 2D arrays and the absolute difference between them.
        The numpy arrays used in the plotting are saved.

        Parameters
        ----------
        grid1 : numpy array
            array to be plotted and compared against grid2.
        grid2 : numpy array
            array to be plotted and compared against grid1.
        vcut : list, optional
            vertical slice of the array (TWT). The default is False.
        hcut : list, optional
            horizontal slice of the array (traces). The default is False.
        cmap : str, optional
            colormap. The default is 'jet'.
        titles : list, optional
            list of titles for the different plots. The default is ['True','Estimate', 'MSE'].
        file_name : str, optional
            file path for the results to be saved to. The default is False.
        extent : list, optional
            axis extent as traces and TWT. The default is [-1,1,-1,1].
        ylab : str, optional
            y label. The default is 'TWT'.
        vmin : float, optional
            minimum value of the plotted array. The default is 0.2.
        vmax : float, optional
            maximum value of the plotted array. The default is 0.4.

        Returns
        -------
        None.

        """
        
        if vcut == False:
            pass
        else:
            TWT = np.linspace(extent[3], extent[2], len(grid1[1, :]))
            extent[3] = TWT[vcut[0]:vcut[1]][0]
            extent[2] = TWT[vcut[0]:vcut[1]][-1]
            
            grid1 = grid1[:, vcut[0]:vcut[1]]
            grid2 = grid2[:, vcut[0]:vcut[1]]
            
        if hcut == False:
            pass
        else:
            grid1 = grid1[hcut[0]:hcut[1],:]
            grid2 = grid2[hcut[0]:hcut[1],:]
        plt.subplot(211)
        
        plt.imshow(grid1.T, cmap=cmap, aspect='auto',vmin=vmin, vmax=vmax, extent=extent)
        plt.xlabel('Trace')
        plt.ylabel(ylab)
        plt.title(titles[0])
        
        plt.subplot(212)
        plt.imshow(grid2.T, cmap=cmap, aspect='auto',vmin=vmin, vmax=vmax, extent=extent) 
        plt.xlabel('Trace')
        plt.ylabel(ylab)
        cax = plt.axes([0.95, 0.1, 0.075, 0.8]) # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_adjust.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-adjust-py
        plt.colorbar(cax=cax)
        if file_name != False:
            np.save(file_name+' grid1', grid1)
            np.save(file_name+' grid2', grid2)
            plt.savefig(file_name+'.png', bbox_inches='tight')
        plt.show()      
        plt.subplot(111)
        uf = usefull_functions()
        sim = uf.my_map2(grid1, grid2, uf.diff)
        plt.imshow(sim.T, cmap='nipy_spectral', aspect='auto', vmin = 0, vmax = 0.15, extent=extent)
        plt.xlabel('Trace')
        plt.ylabel(ylab)
        plt.title(titles[2])
        cax = plt.axes([0.95, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)
        if file_name != False:
            np.save(file_name+' difference', sim)
            plt.savefig(file_name+' difference plot.png', bbox_inches='tight')
        plt.show()
        