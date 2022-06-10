# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:00:06 2021

@author: Eier
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import sklearn
from sklearn.model_selection import KFold 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from classes.load_well import load_well
from sklearn import linear_model


class Mlearning:
    """
    Handels the cross validation, training and application (predictiion)
    of the machine learning methods.
    """
    def __init__(self, pred, response, grid):
        """
        Handels the cross validation, training and application (predictiion)
        of the machine learning methods.

        Parameters
        ----------
        pred : numpy array
            predictors
        response : numpy array
            response or target values
        grid : numpy array
            grid to map the predictions on, for the thesis: the impedance section

        Returns
        -------
        None.

        """
        # set the seeds, means that the result will always be the same given the same data and parameters
        np.random.seed(seed = 1)
        tf.random.set_seed(1)
        
        self.pred =pred
        self.response=response
        self.grid=grid
        
    def KNN_init(self, k):
        """
        Initiallise and fit KNN

        Parameters
        ----------
        k : int
            number of neigbors considered

        Returns
        -------
        None.

        """
        self.model = KNeighborsRegressor(n_neighbors=k)
        self.model.fit(self.pred, self.response)
        
    def RF_init(self, n_estimators = 100,max_depth = 3):
        """
        initiallise and fit random forest

        Parameters
        ----------
        n_estimators : int, optional
            number of trees. The default is 100.
        max_depth : int, optional
            max depth of all decision trees. The default is 3.

        Returns
        -------
        None.

        """
        self.model = RandomForestRegressor(n_estimators = n_estimators,max_depth=max_depth)
        response = np.ravel(self.response)
        self.model.fit(self.pred, response)
        
        
    def lasso_init(self, alpha = 0.1):
        """
        initialise and fit lasso

        Parameters
        ----------
        alpha : float, optional
            coefficiant for the penalty term. The default is 0.1.

        Returns
        -------
        None.

        """
        self.model = linear_model.Lasso(alpha = alpha)
        self.model.fit(self.pred, self.response)
        
    def n_net_init(self, pred, response, n, layers = 1,epochs=10, activation='sigmoid', optimizer = 'adam', verbose=0):
        """
        initialise and fit neural network

        Parameters
        ----------
        pred : numpy array
            predictors
        response : numpy array
            response or target values
        n : int
            number of neurans in each layer.
        layers : int, optional
            number of hidden layers. The default is 1.
        epochs : int, optional
            number of forward and backward prop. The default is 10.
        activation : str, optional
            transfer function. The default is 'sigmoid'.
        optimizer : str, optional
            optimizer function. The default is 'adam'.
        verbose : int, optional
            if 0: stops the network printouts, if 1 the printouts are enabaled. The default is 0.

        Returns
        -------
        None.

        """
        # self.n_model = tf.keras.models.Sequential([
        #   tf.keras.Input(np.shape(self.pred[0])),
        #   tf.keras.layers.Dense(n, activation=activation),
        #   tf.keras.layers.Dense(1, activation=activation)
        # ])
        
        self.model = tf.keras.models.Sequential()
        # add input layer
        self.model.add(tf.keras.Input(np.shape(self.pred[0])))
        
        # add hidden layers
        for i in range(layers):
            self.model.add(tf.keras.layers.Dense(n, activation=activation))
            
        # add output layer
        self.model.add(tf.keras.layers.Dense(1, activation=activation))
        
        self.model.compile(optimizer=optimizer,
                             loss='MeanSquaredError',#'mean_squared_logarithmic_error',#'categorical_crossentropy',#'MeanSquaredError',
                              #loss='MeanAbsoluteError',
                              metrics=['mse'])
        
        self.model.fit(self.pred, self.response, epochs=epochs, verbose=0)
        
        
    def RF_cross_val(self, n_estimators = np.arange(100, 250, 1),max_depth = np.arange(2, 5, 1), cv = 10, plot = True):
        """
        Random forest cross validation

        Parameters
        ----------
        n_estimators : numpy array, optional
            number of decision trees. The default is np.arange(100, 250, 1).
        max_depth : numpy array, optional
            max depth for each decision tree. The default is np.arange(2, 5, 1).
        cv : int, optional
            number of folds. The default is 10.
        plot : bool, optional
            should the cross validation scores the plotted by tuning parameter. The default is True.

        Returns
        -------
        res : dictionary
            resutls of the cross validation by the tuning parameters.

        """
        # prepare to store data
        res = dict()
        i = 0
        scores = np.zeros(len(n_estimators)*len(max_depth))
        estimators_scores = np.zeros(len(n_estimators))
        
        # for all parameters in range
        for k, n in enumerate(n_estimators):
            depth_scores = np.zeros(len(max_depth))
            for l, d in enumerate(max_depth):
                print(i)
                n = int(n)
                
                RF = RandomForestRegressor(n_estimators=n,max_depth=d)
                
                # perform cross validation
                score = cross_val_score(RF, self.pred, self.response, cv = cv, scoring='neg_mean_squared_error').mean()
                
                # store data
                scores[i] = score
                depth_scores[l] = score
                i = i+1
                res['n_estimators: /{}/, max_depth: /{}/'.format(n,d)] = score
            estimators_scores[k] = depth_scores.mean()
            
        # plot cross validation result
        if plot == True:
            plt.plot(max_depth, depth_scores)
            plt.xlabel('max depth')
            plt.ylabel('scores')
            plt.title('CV scores RF')
            plt.show()
            
            plt.plot(n_estimators, estimators_scores)
            plt.xlabel('n_estimators')
            plt.ylabel('scores')
            plt.title('CV scores RF')
            plt.show()
        return res
    
        
    def KNN_cross_val(self, N = np.arange(100, 250, 1), cv = 10, plot = True):
        """
        KNN cross validation

        Parameters
        ----------
        N : numpy array, optional
            number of neigbors considered. The default is np.arange(100, 250, 1).
        cv : int, optional
            number of folds. The default is 10.
        plot : bool, optional
            should the cross validation scores the plotted by tuning parameter. The default is True.

        Returns
        -------
        res : dictionary
            resutls of the cross validation by the tuning parameters.

        """
        # prepare to store data
        res = dict()
        i = 0
        scores = np.zeros(len(N))
        
        # for all parameters in range
        for n in N:
            print(i)
            n = int(n)
            KNN = KNeighborsRegressor(n_neighbors=n)
            
            # perform cross validation
            score = cross_val_score(KNN, self.pred, self.response, cv = cv, scoring='neg_mean_squared_error').mean()
            
            # store data
            scores[i] = score
            i = i+1
            res['k: /{}/'.format(n)] = score
            
        # plot cross validation result
        if plot == True:
            plt.plot(N, scores)
            plt.xlabel('k')
            plt.ylabel('scores')
            plt.title('CV scores KNN')
            plt.show()
        return res
    
    def lasso_cross_val(self, al_array = np.arange(0.01, 10, 0.01), cv = 10, plot = True):
        """
        Lasso-regression cross validation

        Parameters
        ----------
        al_array : numpy array, optional
            penalty coefficients considered. The default is np.arange(0.01, 10, 0.01).
        cv : int, optional
            number of folds. The default is 10.
        plot : bool, optional
            should the cross validation scores the plotted by tuning parameter. The default is True.

        Returns
        -------
        res : dictionary
            resutls of the cross validation by the tuning parameters.

        """
        # prepare to store data
        res = dict()
        i = 0
        scores = np.zeros(len(al_array))
        
        # for all parameters in range
        for alpha in al_array:
            print(i)
            lasso = linear_model.Lasso(alpha = alpha)
            
            # perform cross validation
            score = cross_val_score(lasso, self.pred, self.response, cv = cv, scoring='neg_mean_squared_error').mean()
            
            # store data
            scores[i] = score
            i = i+1
            res['alpha: /{}/'.format(alpha)] = score
            
            
        # plot cross validation result
        if plot == True:
            plt.plot(al_array, scores)
            plt.xlabel('alpha')
            plt.ylabel('scores')
            plt.title('CV scores lasso')
            plt.show()
        return res
        
    def n_net_cross_val(self, N = np.arange(10, 500, 100),layers_list = [1,2, 3], act_list = ['sigmoid'], cv = 10,epochs=10, plot = True):
        """
        Neural Network cross validation

        Parameters
        ----------
        N : numpy array, optional
            range of number of neurans in every layer considered. The default is np.arange(10, 500, 100).
        layers_list : list, optional
            range of number of hidden layers considered. The default is [1,2, 3].
        act_list : list, optional
            activation functions considered. The default is ['sigmoid'].
        cv : int, optional
            number of folds. The default is 10.
        epochs : int, optional
            numnber of epochs or cycles of training. The default is 10.
        plot : bool, optional
            should the cross validation scores the plotted by tuning parameter.

        Returns
        -------
        res : dictionary
            resutls of the cross validation by the tuning parameters.

        """
        
        # prepare to store data
        res = dict()
        
        
        kf = KFold(n_splits=cv)
        
        # for all parameters in range
        for act in act_list:
            layers_scores = np.zeros(len(layers_list))
            
            for l, layers in enumerate(layers_list):
                N_scores = np.zeros(len(N))
                
                for m, n in enumerate(N):
                    i = 0
                    scores = np.zeros(cv)
                    
                    # perform cross validation
                    # split the data and loop for each split
                    for train, test in kf.split(self.pred, self.response): 
                        
                        self.n_net_init(self.pred[train], self.response[train],n = n, layers = layers,activation = act,epochs=epochs)
                        
                        # store data
                        scores[i] = self.model.evaluate(self.pred[test], self.response[test], verbose=0)[0]
                        i+=1
                    
                    # store data
                    score = scores.mean()
                    N_scores[m] = score
                    res['layers: /{}/, activation: /{}/, neurons: /{}/'.format(layers,act, n)] = scores.mean()
                    tf.keras.backend.clear_session()
                layers_scores[l] = N_scores.mean()
                
            # plot cross validation result
            if plot == True:
                plt.plot(N, N_scores)
                plt.xlabel('neurons')
                plt.ylabel('scores')
                plt.title('CV scores N-net')
                plt.show()
                
                plt.plot(np.array(layers_list), layers_scores)
                plt.xlabel('layers')
                plt.ylabel('scores')
                plt.title('CV scores N-net')
                plt.show()

        return res
        
    
    def predict_syn_grid(self,grid_list,grid_names, win, method = 'KNN', grid_ext = 'case1', geo_int = 'none',
                      w_start = [267, 178], w_end_top = [0, 178], w_end_base = [0, 273], 
                      max_TWT = -618, min_TWT = -1162,horizons_list = None,
                      mean_arr = None, std_arr = None):
        """
        Use Trained ML model to predict the target values of the synthetic cross-section (grid)

        Parameters
        ----------
        grid_list : list
            list of 2d arrays representing the cross-sections.
        grid_names : list
            list of grid names, for example ['imp', 'por'].
        win : int
            window size for mean and median rolling window and window selection.
        method : str, optional
            machine learning method. The default is 'KNN'.
        grid_ext : str, optional
            predictor extraction preset. The default is 'case1'.
        geo_int : str, optional
            Should the depositional time be implemented and if so how?. The default is 'none'.
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
        horizons_list : list or None, optional
            list of numpy arrays that describe where the horizons intersect with the cross-section. The default is None.
        mean_arr : float, optional
            the mean value used in the standardization. The default is None.
        std_arr : float, optional
            the standard deviation used in the standardization. The default is None.

        Returns
        -------
        pred_map : numpy array
            result of the prediction.

        """
        
        v, h = np.shape(self.grid)
        # reserve memory for result
        pred_map = np.zeros((v, h-win*2))
        # plt.imshow(self.grid.T)
        # plt.show()
        
        # for every trace
        for col in range(len(self.grid)):
            print('col = ', col)
            # c = self.grid[col]
            # data = pd.DataFrame(c, columns = ['AI'])
            
            ################### Load well ####################################
            log1 = load_well(file_name = 'data_2d_wedge_F3\F03_2_por_eff.xlsx') # arbitrary file name
            data = log1.from_synthetic(grid_list, grid_names, col = col)
            data.drop('Por', axis=1, inplace=True)
            ###################################
            
            # predictor extraction
            from classes.predictor_ext import predictor_ext
            new_pred = predictor_ext(data)
            name = 'imp'
            # new_pred.add_well_loc(well_loc=col)
            #######################################################
            if grid_ext == 'case1':
                new_pred.roll_mean(data_name = name,win = win)
                new_pred.roll_median(data_name = name,win = win)
                new_pred.win_select(data_name = name,win = win)
                
            elif grid_ext == 'case2':
                new_pred.roll_mean(data_name = name,win = win)
                new_pred.roll_median(data_name = name,win = win)
            else:
                pass
            #######################################################
            
            if geo_int == 'wedge':
                new_pred.construct_timelines_wedge_df(self.grid,  w_start = w_start, w_end_top=w_end_top, w_end_base=w_end_base, 
                                                      col = col, win = win)
            elif geo_int == 'from horizons':
                new_pred.construct_timelines_from_horizons(self.grid, horizons_list, max_TWT = max_TWT, min_TWT = min_TWT)
                new_pred.depotime_well(well_loc = col)
                
            #######################################################
            

            new_pred.remove_outside_window(win = win) 
            self.new_pred = new_pred
            
            pred = new_pred.data.to_numpy()
            
            # standardize predictor based on previous standardization
            if type(mean_arr)==np.ndarray and type(std_arr) ==np.ndarray:
                pred, dummy1, dummy2 = new_pred.stand_pred(pred,mean_list = mean_arr, std_list = std_arr)
            else:
                pass
            
            # predict
            prediction = self.model.predict(pred)
            #######################################################
            if method == 'RF' or method == 'lasso':
                pred_map[col] = prediction
            else:
                pred_map[col] = prediction[:,0]
            #######################################################
        return pred_map                            
    
    def predict_grid(self, grid_list,grid_names, win, method = 'KNN', grid_ext = 'case1', geo_int = 'none',
                      horizons_list = None, max_TWT = -618, min_TWT = -1162,
                      mean_arr = None, std_arr = None):
        """
        Use Trained ML model to predict the target values of the cross-section (grid)

        Parameters
        ----------
        grid_list : list
            list of 2d arrays representing the cross-sections.
        grid_names : list
            list of grid names, for example ['imp', 'por'].
        win : int
            window size for mean and median rolling window and window selection.
        method : str, optional
            machine learning method. The default is 'KNN'.
        grid_ext : str, optional
            predictor extraction preset. The default is 'case1'.
        geo_int : str, optional
            Should the depositional time be implemented and if so how?. The default is 'none'.
        horizons_list : list or None, optional
            list of numpy arrays that describe where the horizons intersect with the cross-section. The default is None.
        max_TWT : int, optional
            maximum value of the TWT. The default is -618.
        min_TWT : int, optional
            minimum value of the TWT. The default is -1162.
        mean_arr : float, optional
            the mean value used in the standardization. The default is None.
        std_arr : float, optional
            the standard deviation used in the standardization. The default is None.

        Returns
        -------
        pred_map : numpy array
            result of the prediction.

        """

        
        v, h = np.shape(self.grid)
        # reserve memory for result
        pred_map = np.zeros((v, h-win*2))
        # plt.imshow(self.grid.T)
        # plt.show()
        
        # for every trace
        for col in range(len(self.grid)):
            print('col = ', col)
            # c = self.grid[col]
            # data = pd.DataFrame(c, columns = ['AI'])
            
            #################### Load well ###################################
            log1 = load_well(file_name = 'data_2d_wedge_F3\F03_2_por_eff.xlsx')
            data = log1.from_synthetic(grid_list, grid_names, col = col)
            # data.drop('Por', axis=1, inplace=True)
            ###################################
            
            # predictor extraction
            from classes.predictor_ext import predictor_ext
            new_pred = predictor_ext(data)
            name = 'imp'
            # new_pred.add_well_loc(well_loc=col)
            #######################################################
            if grid_ext == 'case1':
                new_pred.roll_mean(data_name = name,win = win)
                new_pred.roll_median(data_name = name,win = win)
                new_pred.win_select(data_name = name,win = win)
                
            elif grid_ext == 'case2':
                new_pred.roll_mean(data_name = name,win = win)
                new_pred.roll_median(data_name = name,win = win)
            else:
                pass
            #######################################################
            if geo_int == 'from horizons':
                new_pred.construct_timelines_from_horizons(self.grid, horizons_list, standardizing = False, max_TWT = max_TWT, min_TWT = min_TWT)
                new_pred.depotime_well(well_loc = col)
                
            #######################################################
            

            new_pred.remove_outside_window(win = win) 
            self.new_pred = new_pred
            
            pred = new_pred.data.to_numpy()
            
            # standardize predictor based on previous standardization
            if type(mean_arr)==np.ndarray and type(std_arr) ==np.ndarray:
                pred, dummy1, dummy2 = new_pred.stand_pred(pred,mean_list = mean_arr, std_list = std_arr)
            else:
                pass
            
            # predict
            prediction = self.model.predict(pred)
            #######################################################
            if method == 'RF' or method == 'lasso':
                pred_map[col] = prediction
            else:
                pred_map[col] = prediction[:,0]
            #######################################################
        return pred_map
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    