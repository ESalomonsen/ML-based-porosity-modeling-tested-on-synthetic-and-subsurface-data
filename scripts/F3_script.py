# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:30:55 2022

@author: Eier
"""
from classes.manage_cases_class import manage_cases_class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# np.random.seed(seed = 1)

case_type = 'real'

case_list =['case F3']

well_paths_horizons = ['data\case F3\F02_1 well path horizon loc.txt', 'data\case F3\F03_2 well path horizon loc.txt']

window_list = [0, 10]

well_files = ['data\case F3\F02_1_new.txt', 'data\case F3\F03_2_new.txt']

geo_int_list = [False, True]
# 'lasso', 'KNN','Random Forest', 'Neural Net'
method_list = ['lasso', 'KNN','Random Forest', 'Neural Net']

for case in case_list:
    MAE_list = []
    MSE_list = []
    Rs_list = []
    CV_MSE_list = []

    case_list_new = []
    wells_list_new = []
    window_list_new = []
    geo_int_list_new = []
    method_list_new = []

    for window in window_list:
        if window == 0:
            well_files = ['data\case F3\F02_1_new.txt', 'data\case F3\F03_2_new.txt']
        else:
            well_files = ['data\case F3\F02_1_new_upsacale.txt', 'data\case F3\F03_2_new_upsacale.txt']
        for geo_int in geo_int_list:
            for method in method_list:
                
                np.random.seed(seed = 1)

                case_class = manage_cases_class(case = case, wells_loc = None, window = window,geo_int = geo_int)
                
                case_class.load_sections(plot = False, scaling = False)
                
                case_class.load_wells_and_predictors_ext(well_files = well_files, well_paths_horizons = well_paths_horizons)
                

                cv_mse = case_class.ML_with_cross_val(method, layers_list = [1],N = np.arange(1, 100, 1), cv = 10, plot = True,
                                      neurons = np.arange(1, 40, 1),
                                      max_depth = np.arange(1,20, 1),n_estimators = np.arange(1, 41, 10), 
                                      al_array = np.arange(0, 0.005, 0.00005))
                
                case_class.predict(case_type = case_type)
                case_class.result_eval_F3()
                
                # case_class.predict(case_type = case_type)
                # plt.imshow(case_class.pred_map.T)
                # plt.show()
                
                # MAE_list.append(MAE)
                # MSE_list.append(MSE)
                # Rs_list.append(Rs)
                CV_MSE_list.append(cv_mse)
                
                case_list_new.append(case)
                # wells_list_new.append(wells_loc)
                window_list_new.append(window)
                geo_int_list_new.append(geo_int)
                method_list_new.append(method)
                    
                # df2 = pd.DataFrame()
                # df2['case'] = case
                # df2['MAE'] = MAE
                # df2['MSE'] = MSE
                # df2['R-squared'] = r_squared
                
                # df2.to_csv('results\dataframes\{} classic solution'.format(case)+'.txt',sep='/', index = False, encoding = 'utf-8')
                        
    df = pd.DataFrame()

    df['case'] = case_list_new
    df['ML method'] = method_list_new
    # df['wells location'] = wells_list_new
    df['window size'] = window_list_new
    df['depo time implementation'] = geo_int_list_new
    df['cross validation MSE'] = CV_MSE_list
    # df['MAE'] = MAE_list
    # df['MSE'] = MSE_list
    # df['R-squared'] = Rs_list

    df.to_csv('results\dataframes\case {} auto results'.format(case)+'.txt',sep='/', index = False, encoding = 'utf-8')