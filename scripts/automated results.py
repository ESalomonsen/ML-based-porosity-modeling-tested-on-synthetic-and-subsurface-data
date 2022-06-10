# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:37:16 2022

@author: Eier
"""

import numpy as np
import pandas as pd

#################################################################################

file = 'results\dataframes\case case 1a Wedge auto results.txt'

case_1a_df = pd.read_csv(file, sep = '/')
case_1a_df = case_1a_df.sort_values('MAE')
case_1a_df = case_1a_df.reset_index(drop = True)

file = 'results\dataframes\case case 1b Wedge auto results.txt'

case_1b_df = pd.read_csv(file, sep = '/')

case_1b_df = pd.read_csv(file, sep = '/')
case_1b_df = case_1b_df.sort_values('MAE')
case_1b_df = case_1b_df.reset_index(drop = True)

file = 'results\dataframes\case case 2a wedge hetero auto results.txt'

case_2a_df = pd.read_csv(file, sep = '/')

case_2a_df = pd.read_csv(file, sep = '/')
case_2a_df = case_2a_df.sort_values('MAE')
case_2a_df = case_2a_df.reset_index(drop = True)

file = 'results\dataframes\case case 2b wedge hetero auto results.txt'

case_2b_df = pd.read_csv(file, sep = '/')

case_2b_df = pd.read_csv(file, sep = '/')
case_2b_df = case_2b_df.sort_values('MAE')
case_2b_df = case_2b_df.reset_index(drop = True)

file = 'results\dataframes\case case 3a fault auto results.txt'

case_3a_df = pd.read_csv(file, sep = '/')

case_3a_df = pd.read_csv(file, sep = '/')
case_3a_df = case_3a_df.sort_values('MAE')
case_3a_df = case_3a_df.reset_index(drop = True)

file = 'results\dataframes\case case 3b fault auto results.txt'

case_3b_df = pd.read_csv(file, sep = '/')

case_3b_df = pd.read_csv(file, sep = '/')
case_3b_df = case_3b_df.sort_values('MAE')
case_3b_df = case_3b_df.reset_index(drop = True)

#################################################################################

file = 'results\dataframes\case 2a wedge hetero classic solution.txt'

case_2a_classic_df = pd.read_csv(file, sep = '/')

file = 'results\dataframes\case 2b wedge hetero classic solution.txt'

case_2b_classic_df = pd.read_csv(file, sep = '/')

file = 'results\dataframes\case 3a fault classic solution.txt'

case_3a_classic_df = pd.read_csv(file, sep = '/')

file = 'results\dataframes\case 3b fault classic solution.txt'

case_3b_classic_df = pd.read_csv(file, sep = '/')