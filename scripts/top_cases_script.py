# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:19:29 2022

@author: Eier
"""
from classes.manage_cases_class import manage_cases_class
import numpy as np
import pandas as pd


# define parameters

# 'case 1a Wedge', 'case 1b Wedge', 'case 2a wedge hetero', 'case 2b wedge hetero', 'case 3a fault', 'case 3b fault'
case_list =[ 'case 2a wedge hetero', 'case 2b wedge hetero', 'case 3a fault', 'case 3b fault']
window_list = [0, 10]
geo_int_list = [False, True]
# 'lasso', 'KNN','Random Forest', 'Neural Net'
method_list = ['lasso', 'KNN','Random Forest', 'Neural Net']


for case in case_list:
    
    # prepare to store data
    MAE_list = []
    MSE_list = []
    Rs_list = []
    CV_MSE_list = []
    AE_std_list = []

    case_list_new = []
    wells_list_new = []
    window_list_new = []
    geo_int_list_new = []
    method_list_new = []
    
    # define wells depending on the model
    if case == 'case 1a Wedge' or case == 'case 1b Wedge':
        wells_loc_list = [[100, 200], [100], [290]]
        vcut= [120, 400]
        hcut = False
        
    else: 
        # [200, 700], [200], [800], [400]
        wells_loc_list = [[400], [200, 350, 500, 700], [700, 750, 650, 775], [750]]
        vcut= False
        hcut = False
    
    # for every combination of the parameters
    for wells_loc in wells_loc_list:
        for window in window_list:
            for geo_int in geo_int_list:
                for method in method_list:
                    # define seed
                    np.random.seed(seed = 1)
                    
                    # initiallise
                    case_class = manage_cases_class(case = case, wells_loc = wells_loc, window = window,geo_int = geo_int)
                    
                    # get cross-sections
                    case_class.load_sections(plot = False)
                    
                    # get wells and perform predictor extraction
                    case_class.load_synthetic_wells_and_predictors_ext()
                    
                    # train machine learning models with cross validation
                    if case == 'case 2a wedge hetero' or case == 'case 2b wedge hetero': # less data due to resolusion means KNN needs a k that will not exceed the number of datapoints
                        cv_mse = case_class.ML_with_cross_val(method, layers_list = [1],N = np.arange(1, 50, 1), cv = 10, plot = True,
                                              neurons = np.arange(1, 40, 1),
                                              max_depth = np.arange(4,9, 1),n_estimators = np.arange(25, 110, 10)
                                              , al_array = np.arange(0, 0.005, 0.00005))
                    else:
                        cv_mse = case_class.ML_with_cross_val(method, layers_list = [1],N = np.arange(1, 100, 1), cv = 10, plot = True,
                                              neurons = np.arange(1, 40, 1),
                                              max_depth = np.arange(4,9, 1),n_estimators = np.arange(25, 110, 10)
                                              , al_array = np.arange(0, 0.005, 0.00005))
                    
                    # predict porosity cross-section
                    case_class.predict()
                    
                    # compare to True porosity
                    MAE, MSE, Rs,AE_std = case_class.result_eval(vcut= vcut,hcut = hcut)
                    
                    # store data
                    MAE_list.append(MAE)
                    MSE_list.append(MSE)
                    Rs_list.append(Rs)
                    CV_MSE_list.append(cv_mse)
                    AE_std_list.append(AE_std)
                    
                    case_list_new.append(case)
                    wells_list_new.append(wells_loc)
                    window_list_new.append(window)
                    geo_int_list_new.append(geo_int)
                    method_list_new.append(method)
                    if case == 'case 1a Wedge' or case == 'case 1b Wedge':
                        pass
                    # compare SGS solution to true porosity, and compile reults 
                    else:
                        
                        MAE, MSE, r_squared, AE_std = case_class.petrel_solution_eval()
                        
                        df2 = pd.DataFrame()
                        df2['case'] = [case]
                        df2['MAE'] = [MAE]
                        df2['MSE'] = [MSE]
                        df2['R-squared'] = [r_squared]
                        df2['abs error std'] = [AE_std]
                        
                        df2.to_csv('results\dataframes\{} classic solution'.format(case)+'.txt',sep='/', index = False, encoding = 'utf-8')
    
    # compile results of ML
    df = pd.DataFrame()

    df['case'] = case_list_new
    df['ML method'] = method_list_new
    df['wells location'] = wells_list_new
    df['window size'] = window_list_new
    df['depo time implementation'] = geo_int_list_new
    df['cross validation MSE'] = CV_MSE_list
    df['MAE'] = MAE_list
    df['MSE'] = MSE_list
    df['abs error std'] =AE_std_list
    df['R-squared'] = Rs_list

    df.to_csv('results\dataframes\case {} auto results'.format(case)+'.txt',sep='/', index = False, encoding = 'utf-8')
                    