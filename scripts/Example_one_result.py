# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:13:05 2022

Example of how to make one predction

@author: Eier
"""
from classes.manage_cases_class import manage_cases_class
import numpy as np

case = 'case 2a wedge hetero'
window = 5
geo_int = True
method = 'Random Forest'
wells_loc = [200, 500]

# start class which sets the basic parameters
case_class = manage_cases_class(case = case, wells_loc = wells_loc, window = window,geo_int = geo_int)
# get the cross-sections
case_class.load_sections()
# load the well-logs and perform predictor extraction
case_class.load_synthetic_wells_and_predictors_ext()
# train the ML model
case_class.ML_with_cross_val(method, max_depth = np.arange(4,9, 1),n_estimators = np.arange(25, 110, 10))
# apply the model
case_class.predict()
# get the prediction:
prediction = case_class.pred_map