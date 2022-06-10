# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:06:26 2021

@author: Eier
"""
import numpy as np
import matplotlib.pyplot as plt
from classes.usefull_functions import usefull_functions
import pandas as pd

class predictor_ext:
    
    def __init__(self, data):
        """
        

        Parameters
        ----------
        data : pandas DataFrame
            predictor data.

        Returns
        -------
        None.

        """
        
        self.data = data
        
    def add_well_loc(self, well_loc):
        """
        adds the well location as a predictor.
        Only works for synthetic models

        Parameters
        ----------
        well_loc : int
            trace (well) location.

        Returns
        -------
        None.

        """
        self.data['well location'] = np.repeat(well_loc, len(self.data))
        
        
    def median_and_mean(self, data_name = 'Por.Eff.',win =4):
        """
        adds the rolling mean, median window results as predictors

        Parameters
        ----------
        data_name : str, optional
            name of the column that the rolling windows should apply to. The default is 'Por.Eff.'.
        win : int, optional
            window size. The default is 4.

        Returns
        -------
        None.

        """
        self.roll_mean(data_name = data_name,win = win)
        self.roll_median(data_name = data_name,win = win)
        self.remove_outside_window(win = win)

    def roll_and_win_sel(self, data_name = 'Por.Eff.',win =4, 
                              geo_int = 'none', horizons_list = None,grid = 1, col = 1,
                              w_start = [267, 178], w_end_top = [0, 178], w_end_base = [0, 273],
                              max_TWT = -618, min_TWT = -1162):
        """
        adds the rolling mean, median and selections window results as predictors.
        also adds the depositional time if apporopriate.
        
        Parameters
        ----------
        data_name : str, optional
            name of the column that the rolling windows should apply to. The default is 'Por.Eff.'.
        win : int, optional
            window size. The default is 4.
        geo_int : str, optional
            Should the depositional time be implemented and if so how?. The default is 'none'.
        horizons_list : TYPE, optional
            DESCRIPTION. The default is None.
        grid : numpy array, optional
            2D array of a cross-section. The default is 1.
        col : int, optional
            trace (well) location. The default is 1.
        w_start : list, optional
            the starting position of the wedge: the pinch point. The default is [267, 178].
        w_end_top : list, optional
            the end of the top surface of the wedge. The default is [0, 178].
        w_end_base : list, optional
            the end of the base surface of the wedge. The default is [0, 273].
        max_TWT : int, optional
            maximum value of the TWT. The default is -618.
        min_TWT : int, optional
            minimum value of the TWT. The default is -1162.

        Returns
        -------
        None.

        """
        
        # self.add_well_loc(well_loc=col)
        
        # add window functions
        self.roll_mean(data_name = data_name,win = win)
        self.roll_median(data_name = data_name,win = win)
        self.win_select(data_name = data_name,win = win)
        
        # add depo-time
        if geo_int == 'wedge':
            self.construct_timelines_wedge_df(grid = grid, w_start = w_start, w_end_top=w_end_top, w_end_base=w_end_base, 
                                                      col = col, win = win)
        elif geo_int == 'from horizons':
            self.construct_timelines_from_horizons(grid, horizons_list, max_TWT = max_TWT, min_TWT = min_TWT)
            self.depotime_well(well_loc = col)
        
        # remove data outside of the windows
        self.remove_outside_window(win = win)
        
    def roll_and_win_sel_well(self, data_name = 'Por.Eff.',win =4, 
                              geo_int = 'none', horizons_list = None,grid = 1, MD = None, well_path = None):
        self.roll_mean(data_name = data_name,win = win)
        self.roll_median(data_name = data_name,win = win)
        self.win_select(data_name = data_name,win = win)
        
        # add depo-time
        if geo_int == 'from horizons':
            self.depotime_well_path(MD = MD, well_path = well_path)
            
        
        self.remove_outside_window(win = win)
##########################################################################################
    def roll_mean(self, data_name = 'Por.Eff.',win =4):
        """
        applies the rolling mean to a column in the dataframe

        Parameters
        ----------
        data_name : str, optional
            name of the column that the rolling windows should apply to. The default is 'Por.Eff.'.
        win : int, optional
            window size. The default is 4.

        Returns
        -------
        data : pandas DataFrame
            full dataframe of the predictors.

        """
        data = self.data
        # failsafe for if no window is wanted
        if win == 0:
            pass
        else:
            data['roll mean'] = data[data_name].rolling(window=win).mean()
            self.data = data
        return data
        
    def roll_median(self, data_name = 'Por.Eff.',win =4):
        """
        applies the rolling median to a column in the dataframe

        Parameters
        ----------
        data_name : str, optional
            name of the column that the rolling windows should apply to. The default is 'Por.Eff.'.
        win : int, optional
            window size. The default is 4.

        Returns
        -------
        data : pandas DataFrame
            full dataframe of the predictors.

        """
        data = self.data
        # failsafe for if no window is wanted
        if win == 0:
            pass
        else:
            data['roll median'] = data[data_name].rolling(window=win).median()
            self.data = data
        return data
        
    def win_select(self, data_name = 'Por.Eff.',win =4):
        """
        applies window selection to a predictor.
        This means that in a window of [-win/2, +win/2] every data point from the point of consideration is added as a predictor

        Parameters
        ----------
        data_name : str, optional
            name of the column that the rolling windows should apply to. The default is 'Por.Eff.'.
        win : int, optional
            window size. The default is 4.

        Returns
        -------
        data : pandas DataFrame
            full dataframe of the predictors.

        """
            
        data = self.data
        # failsafe for if no window is wanted
        if win == 0:
            pass
        else:
            data_col= data[data_name].to_numpy()
            
            
            win_2 = int(win/2)
            l = len(data_col)
            
            # for every point above and below the point of computation
            for w in range(win_2):
                w = w+1
                
                # prepare to get higher value
                up = np.zeros(l)
                up[:] = np.nan
                # get value
                up[w:l] = data_col[0:l-w]
                
                # prepare to get lower value
                down = np.zeros(l)
                down[:] = np.nan
                # get value
                down[0:l-w] = data_col[0+w:l]
                
                # save values
                data['+{} wind'.format(w)] = up
                data['-{} wind'.format(w)] = down
                
            self.data = data
        return data
    
    def remove_outside_window(self, win):
        """
        Remove the top and bottom rows equal to the window size. 

        Parameters
        ----------
        win : int
            window size.

        Returns
        -------
        None.

        """
        
        self.data = self.data.iloc[win:]
        self.data = self.data.iloc[0:len(self.data)-win]

    
    def construct_timelines_wedge(self, grid, w_start = [267, 178], w_end_top = [0, 178], w_end_base = [0, 273]):
        """
        A non-ideal function that constructs an array of the depositional time for a wedge that is thinning to the right.

        Parameters
        ----------
        grid : numpy array
            2D array of a cross-section.
        w_start : list, optional
            the starting position of the wedge: the pinch point. The default is [267, 178].
        w_end_top : list, optional
            the end of the top surface of the wedge. The default is [0, 178].
        w_end_base : list, optional
            the end of the base surface of the wedge. The default is [0, 273].

        Returns
        -------
        res : numpy array
            2D array with the relative depositional time.

        """
        
        inter = 1
        if w_start > w_end_top:
            direction = 'thinning right'
        
        rows = np.arange(min([w_start[0], w_end_top[0]]), max([w_start[0], w_end_top[0]]), 1) #  what rows are relavant
        
        if direction == 'thinning right':
            # start_top0 = np.linspace(),len(rows)).round()
            start_top1 = np.linspace(min(w_start[1], w_end_top[1]),max(w_start[1], w_end_top[1]),len(rows)).round()
        
            # start_base0 = np.linspace(min(w_start[0], w_end_base[0]),max(w_start[0], w_end_base[0]),len(rows)).round()
            start_base1 = np.linspace(max(w_start[1], w_end_base[1]),min(w_start[1], w_end_base[1]),len(rows)).round()
            
        else:
            start_top0 = np.linspace(min(w_start[0], w_end_top[0]),max(w_start[0], w_end_top[0]),len(rows)).round()
            start_top1 = np.linspace(min(w_start[1], w_end_top[1]),max(w_start[1], w_end_top[1]),len(rows)).round()
            
            start_base0 = np.linspace(min(w_start[0], w_end_base[0]),max(w_start[0], w_end_base[0]),len(rows)).round()
            start_base1 = np.linspace(min(w_start[1], w_end_base[1]),max(w_start[1], w_end_base[1]),len(rows)).round()
        
        top_ts = np.arange(0, int(start_top1[1]), inter) # timelines above wedge
        
        missing = np.arange(int(start_top1[1]), int(max(start_base1)), inter) # timelines inside wedge
        
        base_ts = np.arange(int(max(start_base1)),len(grid[0]), inter) # timelines below wedge
        
        L = abs(w_start[0]-w_end_top[0])
        
        change_in_inter = np.linspace(inter, 0, L)
        
        res = np.zeros(np.shape(grid))
        
        for n, row in enumerate(grid): # vertical, left to right in imshow
            print(n)
            l = len(row)
            new_row = np.zeros(l)
        
            if n in rows:
                if direction == 'thinning right':
                    
                    change = change_in_inter[n]
                    
                    new_row[0:int(start_top1[n])] = np.linspace(top_ts[0],top_ts[-1], len(rows[0:int(start_top1[n])]))
                    ll = len(row[int(start_top1[n]):int(start_base1[n])])
                    new_row[int(start_top1[n]):int(start_base1[n])] = np.linspace(missing[0], missing[-1], len(new_row[int(start_top1[n]):int(start_base1[n])]))
                    new_row[int(start_base1[n]):] = np.linspace(base_ts[0], base_ts[0]+len(new_row[int(start_base1[n]):]), len(new_row[int(start_base1[n]):]))
                    
            else:
                new_row[0:int(start_top1[0])] = np.linspace(top_ts[0], top_ts[-1], len(new_row[0:int(start_top1[0])]))
                new_row[int(start_top1[0]):] = np.linspace(base_ts[0], base_ts[0]+len(new_row[int(start_top1[0]):]), len(new_row[int(start_top1[0]):] ))
        
            res[n] = new_row
        
        res = res.round()
        # res = (res - np.mean(res)) / np.std(res) # standardizing data
        
        return res
    
    def construct_timelines_wedge_df(self,  grid,  w_start = [267, 178], w_end_top = [0, 178], w_end_base = [0, 273], 
                                     col = 100):
        """
        Make the relative depositional time for a synthetic wedge,
        then exctract one of the columns as a predictor.

        Parameters
        ----------
        grid : numpy array
            2D array of a cross-section.
        w_start : list, optional
            the starting position of the wedge: the pinch point. The default is [267, 178].
        w_end_top : list, optional
            the end of the top surface of the wedge. The default is [0, 178].
        w_end_base : list, optional
            the end of the base surface of the wedge. The default is [0, 273].
        col : int, optional
            trace (well) location. The default is 100.

        Returns
        -------
        self.data : pandas DataFrame
            full dataframe of the predictors.

        """
        
        # get the timelines
        self.timelines = self.construct_timelines_wedge(grid, w_start, w_end_top, w_end_base)
        
        data = self.timelines[col]
        
        # save the timelines for one trace
        self.data['time lines {}'.format(col)] = data
        
        # plt.imshow(self.timelines.T, aspect = 'auto',cmap='jet')
        # plt.title('Depotime from wedge interpretaion')
        # plt.colorbar()
        # plt.show()
        
        
        # self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        return self.data
    def construct_timelines_from_horizons(self, grid, horizons_list, max_TWT = -618, min_TWT = -1162, standardizing = True):
        """
        make the deopsitional time array from where the horizons interect with the cross-sections

        Parameters
        ----------
        grid : numpy array
            2D array of a cross-section.
        horizons_list : list or None, optional
            list of numpy arrays that describe where the horizons intersect with the cross-section. The default is None.
        max_TWT : int, optional
            maximum value of the TWT. The default is -618.
        min_TWT : int, optional
            minimum value of the TWT. The default is -1162.
        standardizing : bool, optional
            should the resulting array be standardized?. The default is True.

        Returns
        -------
        res : numpy array
            2D array of the depositional time.

        """
        # ordered from max to min
        n_traces, n_samples= np.shape(grid)
        interval = -abs((max_TWT-min_TWT)/n_samples)
        print(n_samples)
        
        res = np.zeros(np.shape(grid))
        
        # for every trace
        for s in range(n_traces):
            stack = np.array([])
            # for every horizon
            for i in range(len(horizons_list)):
                # find the upper and lower horizons
                upper_h = horizons_list[i-1]
                lower_h = horizons_list[i]
                
                # firs (top) horizon
                if i == 0:
                    # if the upper horizon extends beyond the cross-section then skip
                    if upper_h[s] > max_TWT:
                        pass
                    # if the upper horizon does no extend beyond the cross-section then fill in the missing values above the horizon
                    else:
                        
                        upper_h = horizons_list[i]
                        times = np.arange(max_TWT, upper_h[s], interval)
                        times = times[times<=max_TWT]
                        
                        times = np.linspace(i, i+1, len(times))
                        
                        stack = np.hstack((stack, times))
                else:

                    times = np.arange(upper_h[s], lower_h[s], interval)
                    
                    times = times[times<=max_TWT]
                    
                    times = np.linspace(i, i+1, len(times))
                    
                    stack = np.hstack((stack, times))
                    
            if len(stack) < len(res[s]): # happens if the bottom horizon is not beyond the min_TWT, meaning the remaining time needs to be filled in
                diff = abs(len(stack) - len(res[s]))
                stack = np.hstack((stack, np.linspace(i+1, i+2, diff)))
                        
            
            res[s] = stack[0:len(res[s])]

            
            self.times = res
            
        # plt.imshow(res.T, aspect = 'auto',cmap='jet')
        # plt.title('Depotime from horizons')
        # plt.colorbar()
        # plt.show()
        
        if standardizing == True:
            pass
            # res = (res - np.mean(res)) / np.std(res) # standardizing data
        # plt.imshow(res.T, cmap='nipy_spectral', aspect='auto', extent=[0,600,min_TWT,max_TWT])
        # plt.colorbar()
        # plt.show()
        return res
    
    def depotime_well_path(self, MD = None, well_path = None):
        """
        Add the depositional time as a predictor.
        This is for a real well, not synthetic data.

        Parameters
        ----------
        MD : numpy array, optional
            measured depth of the well-logs. The default is None.
        well_path : str, optional
            path to a file containing the horizon locations intersecting the cross-section. The default is None.

        Returns
        -------
        None.

        """
        
        MD = -MD
        
        # get the horizon locations in the well
        well_p = pd.read_csv(well_path)
        well_p_np = -well_p.to_numpy()
        
        well_p_np.sort()
        well_p_np = -well_p_np[0]
        
        uf = usefull_functions()
        
        indx_list = []
        
        # for each horizon
        for hor_loc in well_p_np:
            
            # find the horizon location in the MD-log
            indx = uf.nearest(hor_loc, MD)[0]
            indx_list.append(indx)
            
        indx_list.sort()
        
        # interpolating the time
        time = np.zeros(len(MD))
        i_old = 0
        n_old = 0
        for n, i in enumerate(indx_list):
            
            time[i_old:i] = np.linspace(n_old, n+1,len(time[i_old:i]) )
            
            i_old = i
            n_old = n+1
            
        time[i_old:] = np.linspace(n_old, n_old+1,len(time[i_old:]) )
        
        self.time = time
        
        self.data['depotime'] = self.time
    
    def depotime_well(self, well_loc):
        self.data['depo time at {}'.format(well_loc)] = self.times[well_loc]
        
    def stand_pred(self, pred, mean_list = None, std_list = None):
        """
        standardize all the predictors and save the standard deviation and mean used for the standardization.

        Parameters
        ----------
        pred : numpy array
            numpy array of predictors.
        mean_list : numpy array, optional
            array of the mean values to be used in the standization. The default is None.
        std_list : numpy array, optional
            array of the standard deviation values to be used in the standization. The default is None.

        Returns
        -------
        pred : numpy array
            numpy array of predictors.
        mean_arr : numpy array
            array of the mean values used in the standization.
        std_arr : numpy array
            array of the standard deviation values used in the standization.

        """
        
        # reserving memory
        mean_arr = np.zeros(len(pred[0]))
        std_arr = np.zeros(len(pred[0]))
        
        # if previous mean and std are to be used 
        if type(mean_list)==np.ndarray and type(std_list) ==np.ndarray:
            for n in range(len(pred[0])):
                pred[:,n] = ( pred[:,n] - mean_list[n]) / std_list[n]
                
        else:
        
            for n in range(len(pred[0])):
                mean_arr[n] = np.mean( pred[:,n])
                std_arr[n] = np.std( pred[:,n])
                
                
                
                pred[:,n] = ( pred[:,n] - np.mean( pred[:,n])) / np.std( pred[:,n])
                
            
            
        return pred, mean_arr, std_arr
    
    
