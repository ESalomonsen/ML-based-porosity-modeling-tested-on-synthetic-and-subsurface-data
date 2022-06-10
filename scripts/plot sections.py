# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:51:25 2022

@author: Eier
"""

from classes.seismic_ext import seismic_ext
from classes.plot_results import plot_results


case_list = ['case 1a', 'case 1b', 'case 2a', 'case 2b', 'case 3a', 'case 3b']
for case in case_list:

    if case == 'case 1a':
        file_imp = 'data\case 1a Wedge\Seis_Inv_Wedge_IIIc.segy'
        ext = seismic_ext(file_imp)
        grid, extent = ext.syn_seismic(plot = False, scaling = False)
        plot_results.plot_generic(grid = grid, xlab = 'trace', ylab = 'TWT', extent = extent, vmin = 2000, vmax = 5000,
                                  vcut= [120, 400], title = 'impedance, homogeneous wedge without noise ({})'.format(case))
        
    elif case == 'case 1b':
        file_imp = 'data\case 1b Wedge\Seis_Noise_Inv_Wedge_IIIc.segy'
        ext = seismic_ext(file_imp)
        grid, extent = ext.syn_seismic(plot = False, scaling = False)
        plot_results.plot_generic(grid, xlab = 'trace', ylab = 'TWT', extent = extent,vmin = 2000, vmax = 5000,
                                  vcut= [120, 400], title = 'impedance, homogeneous wedge with noise ({})'.format(case))
    
    elif case == 'case 2a':
        file_imp = 'data\case 2a wedge hetero\case 2a wedge imp no noise.segy'
        ext = seismic_ext(file_imp)
        grid, extent = ext.syn_seismic(plot = False, scaling = False)
        plot_results.plot_generic(grid, xlab = 'trace', ylab = 'TWT', extent = extent, vmin = 3500000, vmax = 5500000,
                                  title = 'impedance, heterogeneous wedge without noise ({})'.format(case))
        
    elif case == 'case 2b':
        file_imp = 'data\case 2b wedge hetero\case 2b wedge imp noise.segy'
        ext = seismic_ext(file_imp)
        grid, extent = ext.syn_seismic(plot = False, scaling = False)
        plot_results.plot_generic(grid, xlab = 'trace', ylab = 'TWT', extent = extent, vmin = 3500000, vmax = 5500000,
                                  title = 'impedance, heterogeneous wedge with noise ({})'.format(case))
        
    elif case == 'case 3a':
        file_imp = 'data\case 3a fault\case 3a fault imp no noise.segy'
        ext = seismic_ext(file_imp)
        grid, extent = ext.syn_seismic(plot = False, scaling = False)
        plot_results.plot_generic(grid, xlab = 'trace', ylab = 'TWT', extent = extent, vmin = 3500000, vmax = 5500000,
                                  title = 'impedance, Normal fault without noise ({})'.format(case))
        
    elif case == 'case 3b':
        file_imp = 'data\case 3b fault\case 3b fault imp noise.segy'
        ext = seismic_ext(file_imp)
        grid, extent = ext.syn_seismic(plot = False, scaling = False)
        plot_results.plot_generic(grid, xlab = 'trace', ylab = 'TWT', extent = extent, vmin = 3500000, vmax = 5500000,
                                  title = 'impedance, Normal fault with noise ({})'.format(case))






# ext = seismic_ext(file_seis)
# grid1 = ext.syn_seismic(plot = True, scaling = True)

# ext = seismic_ext(file_ai)
# grid2 = ext.syn_seismic(plot = True,scaling = False)

# file_ai = 'data\case 1b Wedge\Seis_Noise_Inv_Wedge_IIIc.segy'
# file_seis = 'data\case 1b Wedge\Synth_Noise_Wedge_IIIc.segy'

# ext = seismic_ext(file_seis)
# grid1 = ext.syn_seismic(plot = True, scaling = True)

# ext = seismic_ext(file_ai)
# grid2 = ext.syn_seismic(plot = True,scaling = True)

# file_ai = 'data\case 2a wedge hetero\case 2 wedge imp.segy'
# file_seis = 'data\case 2a wedge hetero\case 2 wedge seis.segy'

# ext = seismic_ext(file_seis)
# grid1 = ext.syn_seismic(plot = True, scaling = True)

# ext = seismic_ext(file_ai)
# grid2 = ext.syn_seismic(plot = True,scaling = True)

# file_ai = 'data\case 2b wedge hetero\case 2 wedge imp.segy'
# file_seis = 'data\case 2b wedge hetero\case 2 wedge seis.segy'

# ext = seismic_ext(file_seis)
# grid1 = ext.syn_seismic(plot = True, scaling = True)

# ext = seismic_ext(file_ai)
# grid2 = ext.syn_seismic(plot = True,scaling = True)

# file_ai = 'data\case 3a fault\case 3 fault imp.segy'
# file_seis = 'data\case 3a fault\case 3 fault seis.segy'

# ext = seismic_ext(file_seis)
# grid1 = ext.syn_seismic(plot = True, scaling = True)

# ext = seismic_ext(file_ai)
# grid2 = ext.syn_seismic(plot = True,scaling = True)

# file_ai = 'data\case 3b fault\case 3 fault imp.segy'
# file_seis = 'data\case 3b fault\case 3 fault seis.segy'

# ext = seismic_ext(file_seis)
# grid1 = ext.syn_seismic(plot = True, scaling = True)

# ext = seismic_ext(file_ai)
# grid2 = ext.syn_seismic(plot = True,scaling = True)