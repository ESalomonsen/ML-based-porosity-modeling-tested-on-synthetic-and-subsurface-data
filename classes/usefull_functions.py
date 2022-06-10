# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:53:33 2022

@author: Eier
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

class usefull_functions:
    
    def __init__(self):
        pass
    
    def save_cv(self, path = 'CV.pkl', method = 'KNN', dictionary = ''):
        
        # save the cv results
        a_file = open(path, "wb")
        pickle.dump(dictionary, a_file)
        a_file.close()
        
        # https://www.tutorialsteacher.com/articles/sort-dict-by-value-in-python
        marklist=sorted((value, key) for (key,value) in dictionary.items())
        sortdict=dict([(k,v) for v,k in marklist])
        self.cv_res = sortdict
        
        # get the best results 
        if method == 'Neural Net':
            best = list(sortdict)[0]
        else:
            best = list(sortdict)[-1]
            
        return best, sortdict[best]
    
        
    def nearest(self, number, arr):
        """
        Gets index of number nearest the "number"
    
        Parameters
        ----------
        number : TYPE
            DESCRIPTION.
        arr : TYPE
            DESCRIPTION.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        """
        search = abs(arr-number)
        m = search.min()
        return np.where(search == m)[0]    
    
    def diff(self, v1, v2):
        return abs(v1-v2)
    
    def my_map(self, grid, func):
        """
        I find the existing map function to be limited, this is an alternitive
        that applies a function to all elements in a grid/matrix.
        """
        n,k = np.shape(grid)
        l = list(grid.flatten())
        m = list(map(func, l))
        a = np.array(m)
        return a.reshape(n, k)
    
    def my_map2(self, grid1, grid2, func, cross = True):
        """
        I find the existing map function to be limited, this is an alternitive
        that applies a function to all elements in a grid/matrix.
        """
        # if cross == True:
        #     mean1 = np.mean(grid1)
        #     std1 = np.std(grid1) 
        #     mean2 = np.mean(grid2)
        #     std2 = np.std(grid2)  
        
        n,k = np.shape(grid1)
        l1 = list(grid1.flatten())
        l2 = list(grid2.flatten())
        m = list(map(func, l1, l2))
        a = np.array(m)
        return a.reshape(n, k)