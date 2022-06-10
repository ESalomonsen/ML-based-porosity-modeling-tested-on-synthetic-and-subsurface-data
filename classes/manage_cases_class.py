# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:43:48 2022

@author: Eier
"""
import numpy as np
import segyio
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.metrics import r2_score

from classes.load_well import load_well
from classes.predictor_ext import predictor_ext
from classes.Mlearning import Mlearning
from classes.plot_results import plot_results
from classes.usefull_functions import usefull_functions
from classes.seismic_ext import seismic_ext


class manage_cases_class:
    """
    Class for connecting the other classes and data.
    This makes it easier to run many different parameter combinations automatically (iteratively).
    """

    def __init__(self, case, wells_loc, window, geo_int=False,  # first line is all that is used for the thesis
                 file_ai=None, file_seis=None, file_por=None, max_TWT=None, min_TWT=None,
                 w_start=None, w_end_top=None, w_end_base=None, horizons_list=None, petrel_solution=None):
        """
        sets the parameters of the case from a preset or defines them here

        Parameters
        ----------
        case : str
            the case identification as to get all the preset parameters.
        wells_loc : list
            the trace locations of the wells.
        window : TYPE
            window size to be used in predictor extraction.
        geo_int : str, optional
            Should the depositional time be implemented and if so how?. The default is False.
        file_ai : str, optional
            path to segy file with impedance. The default is None.
        file_seis : str, optional
            path to segy file with seismic. The default is None.
        file_por : str, optional
            path to segy file with porosity. The default is None.
        max_TWT : int, optional
            maximum value of the TWT. The default is None.
        min_TWT : int, optional
            minimum value of the TWT. The default is None.
         w_start : list, optional
             the starting position of the wedge: the pinch point. The default is None.
        w_end_top : list, optional
           the end of the top surface of the wedge. The default is None.
        w_end_base : list, optional
            the end of the base surface of the wedge. The default is None.
        horizons_list : list or None, optional
            list of numpy arrays that describe where the horizons intersect with the cross-section. The default is None.
        petrel_solution : str, optional
            file path to the SGS solution of porosity. The default is None.

        Returns
        -------
        None.

        """

        self.case = case
        self.win = window
        self.wells_loc = wells_loc
        self.geo_int = geo_int

        ###################################
        # for new case:
        self.file_ai = file_ai
        self.file_seis = file_seis
        self.file_por = file_por
        self.max_TWT = max_TWT
        self.min_TWT = min_TWT
        self.w_start = w_start
        self.w_end_top = w_end_top
        self.w_end_base = w_end_base
        self.horizons_list = horizons_list
        self.petrel_solution = petrel_solution

        ###################################
        # select the relevant files based on the case
        if case == 'case 1a Wedge':
            self.file_ai = 'data\case 1a Wedge\Seis_Inv_Wedge_IIIc.segy'
            #self.file_seis = 'data\case 1a Wedge\Synth_Wedge_IIIc.segy'
            self.file_por = 'data\case 1a Wedge\Por_Wedge_IIIc.segy'
            # self.max_TWT = -618 # max TWT of the seismic section
            # self.min_TWT = -1162 # min TWT of the seismic section
            # wedge start, the relitive location where the wedge has just thinned out
            self.w_start = [267, 178]
            # the relitive location where the top of the wedge ends
            self.w_end_top = [0, 178]
            # the relitive location where the base of the wedge ends
            self.w_end_base = [0, 400]
            self.horizons_list = False

            if self.geo_int == True:
                self.geo_int = 'wedge'

        elif case == 'case 1b Wedge':
            self.file_ai = 'data\case 1b Wedge\Seis_Noise_Inv_Wedge_IIIc.segy'
            #self.file_seis = 'data\case 1b Wedge\Synth_Noise_Wedge_IIIc.segy'
            self.file_por = 'data\case 1b Wedge\Por_Wedge_IIIc.segy'
            # self.max_TWT = -618 # max TWT of the seismic section
            # self.min_TWT = -1162 # min TWT of the seismic section
            # wedge start, the relitive location where the wedge has just thinned out
            self.w_start = [267, 178]
            # the relitive location where the top of the wedge ends
            self.w_end_top = [0, 178]
            # the relitive location where the base of the wedge ends
            self.w_end_base = [0, 400]
            self.horizons_list = False

            if self.geo_int == True:
                self.geo_int = 'wedge'

        elif self.case == 'case 2a wedge hetero':
            self.file_ai = 'data\case 2a wedge hetero\case 2a wedge imp no noise.segy'
            #self.file_seis = 'data\case 2a wedge hetero\case 2 wedge seis.segy'
            self.file_por = 'data\case 2a wedge hetero\case 2a wedge porosity.segy'
            self.petrel_solution = 'data\case 2a wedge hetero\case 2a wedge porosity estimation no noise.segy'

            hor_file4 = 'data\case 2a wedge hetero\horizons\wedge surface base plus'
            hor_file3 = 'data\case 2a wedge hetero\horizons\wedge surface base'
            hor_file2 = 'data\case 2a wedge hetero\horizons\wedge surface top'
            hor_file1 = 'data\case 2a wedge hetero\horizons\wedge surface top plus'

            self.horizons_list = [np.load(hor_file1+'TWT'+'.npy'), np.load(
                hor_file2+'TWT'+'.npy'), np.load(hor_file3+'TWT'+'.npy'), np.load(hor_file4+'TWT'+'.npy')]
            # self.max_TWT = -618
            # self.min_TWT = -1162

            if self.geo_int == True:
                self.geo_int = 'from horizons'

        elif self.case == 'case 2b wedge hetero':
            self.file_ai = 'data\case 2b wedge hetero\case 2b wedge imp noise.segy'
            #self.file_seis = 'data\case 2b wedge hetero\case 2 wedge seis.segy'
            self.file_por = 'data\case 2b wedge hetero\case 2b wedge porosity.segy'
            self.petrel_solution = 'data\case 2b wedge hetero\case 2b wedge porosity estimation noise.segy'

            hor_file4 = 'data\case 2b wedge hetero\horizons\wedge surface base plus'
            hor_file3 = 'data\case 2b wedge hetero\horizons\wedge surface base'
            hor_file2 = 'data\case 2b wedge hetero\horizons\wedge surface top'
            hor_file1 = 'data\case 2b wedge hetero\horizons\wedge surface top plus'

            self.horizons_list = [np.load(hor_file1+'TWT'+'.npy'), np.load(
                hor_file2+'TWT'+'.npy'), np.load(hor_file3+'TWT'+'.npy'), np.load(hor_file4+'TWT'+'.npy')]
            # self.max_TWT = -618
            # self.min_TWT = -1162

            if self.geo_int == True:
                self.geo_int = 'from horizons'

        elif self.case == 'case 3a fault':
            self.file_ai = 'data\case 3a fault\case 3a fault imp no noise.segy'
            #self.file_seis = 'data\case 3a fault\case 3 fault seis.segy'
            self.file_por = 'data\case 3a fault\case 3a fault porosity.segy'
            self.petrel_solution = 'data\case 3a fault\case 3a fault porosity no noise.segy'

            hor_file1 = 'data\case 3a fault\horizons\cfault surface 1'
            hor_file2 = 'data\case 3a fault\horizons\cfault surface 4'
            hor_file3 = 'data\case 3a fault\horizons\cfault surface 3'
            hor_file4 = 'data\case 3a fault\horizons\cfault surface 5'
            hor_file5 = 'data\case 3a fault\horizons\cfault surface 2'
            self.horizons_list = [np.load(hor_file1+'TWT'+'.npy'), np.load(hor_file2+'TWT'+'.npy'), np.load(
                hor_file3+'TWT'+'.npy'), np.load(hor_file4+'TWT'+'.npy'), np.load(hor_file5+'TWT'+'.npy')]
            # self.max_TWT = -618
            # self.min_TWT = -1162

            if self.geo_int == True:
                self.geo_int = 'from horizons'

        elif self.case == 'case 3b fault':
            self.file_ai = 'data\case 3b fault\case 3b fault imp noise.segy'
            #self.file_seis = 'data\case 3b fault\case 3 fault seis.segy'
            self.file_por = 'data\case 3b fault\case 3b fault porosity.segy'
            self.petrel_solution = 'data\case 3b fault\case 3b fault porosity estimation noise.segy'
            hor_file1 = 'data\case 3b fault\horizons\cfault surface 1'
            hor_file2 = 'data\case 3b fault\horizons\cfault surface 4'
            hor_file3 = 'data\case 3b fault\horizons\cfault surface 3'
            hor_file4 = 'data\case 3b fault\horizons\cfault surface 5'
            hor_file5 = 'data\case 3b fault\horizons\cfault surface 2'
            self.horizons_list = [np.load(hor_file1+'TWT'+'.npy'), np.load(hor_file2+'TWT'+'.npy'), np.load(
                hor_file3+'TWT'+'.npy'), np.load(hor_file4+'TWT'+'.npy'), np.load(hor_file5+'TWT'+'.npy')]
            # self.max_TWT = -618
            # self.min_TWT = -1162

            if self.geo_int == True:
                self.geo_int = 'from horizons'

        elif self.case == 'case F3':
            self.file_ai = 'data\case F3\Seis Inv depth Random line [2D Converted].segy'
            self.petrel_solution = 'data\case F3\F3 porosity estimation.segy'

            hor_file1 = 'data\case F3\F3-Horizon-FS8 (Z)'
            hor_file2 = 'data\case F3\F3-Horizon-Truncation (Z)'
            hor_file3 = 'data\case F3\F3-Horizon-MFS4 (Z)'

            self.horizons_list = [np.load(hor_file1+'TWT'+'.npy'), np.load(
                hor_file2+'TWT'+'.npy'), np.load(hor_file3+'TWT'+'.npy')]

            if self.geo_int == True:
                self.geo_int = 'from horizons'

        else:
            pass

    def handle_CV(self, name, CV):
        """
        handles cross-validation result

        Parameters
        ----------
        name : str
            file path to save the CV result.
        CV : dict
            dictionary of cross-validation results.

        Returns
        -------
        best : str
            string of the best parameters.
        best_ : list
            list of the best parameters.

        """
        uf = usefull_functions()
        best_, self.neg_mse = uf.save_cv(name + ".pkl", self.ML_method, CV)
        best = best_.split('/')
        file = open(name + ".pkl", "rb")
        CV_KNNoutput = pickle.load(file)

        return best, best_

    def load_sections(self, plot=True, scaling=False):
        """
        load the cross-sections

        Parameters
        ----------
        plot : str, optional
            if true plot the sections. The default is True.
        scaling : str, optional
            should the sections be standardized (not working currently). The default is False.

        Returns
        -------
        None.

        """

        # get impedance
        ext = seismic_ext(self.file_ai)
        self.grid2, self.extent2 = ext.syn_seismic(plot=plot, scaling=scaling)

        # get porosity
        if self.file_por == None:
            pass
        else:
            ext = seismic_ext(self.file_por)
            self.grid3, self.extent3 = ext.syn_seismic(
                plot=plot, scaling=False)

        # get petrel solution

        if self.petrel_solution == None:
            pass
        else:
            ext = seismic_ext(self.petrel_solution)
            self.grid_pet, self.extent_pet = ext.syn_seismic(
                plot=plot, scaling=False)

            if self.case == 'case F3':

                self.grid_pet = np.flip(self.grid_pet, 0)

        # get max and min depth
        self.max_TWT = self.extent2[3]
        self.min_TWT = self.extent2[2]

        # slice the sections based on the case
        if self.case == 'case 3a fault' or self.case == 'case 3b fault':
            self.grid2 = self.grid2[:, 0:135]
        elif self.case == 'case 2a wedge hetero' or self.case == 'case 2b wedge hetero':
            # self.grid1 = self.grid1[:, 0:75]
            self.grid2 = self.grid2[:, 0:75]
            self.grid3 = self.grid3[:, 0:75]

    def load_wells_and_predictors_ext(self, well_files, win_names=['imp'], well_paths_horizons=None):
        """
        load the well-logs as predictors and target, then perform predictor extraction
        on the real data-set

        Parameters
        ----------
        well_files : TYPE
            DESCRIPTION.
        win_names : TYPE, optional
            DESCRIPTION. The default is ['imp'].
        well_paths_horizons : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if well_paths_horizons == None:
            well_paths_horizons = np.linspace(0, 1, len(well_files))

        grid_list = [self.grid2]
        self.grid_list = grid_list
        grid_names = ['imp']
        self.grid_names = grid_names

        well = well_files[0]
        log1 = load_well(file_name=well)

        data, MD = log1.from_csv()

        Por = data['por'].to_numpy()
        Por_df = pd.DataFrame(Por, columns=['Por'])

        data.drop('por', axis=1, inplace=True)

        new_pred = predictor_ext(data)

        for name in win_names:
            new_pred.roll_and_win_sel_well(data_name=name, win=self.win,
                                           geo_int=self.geo_int, grid=self.grid2, horizons_list=self.horizons_list, MD=MD, well_path=well_paths_horizons[0])
        new_pred2 = predictor_ext(Por_df)
        new_pred2.remove_outside_window(win=self.win)

        data1 = new_pred2.data
        data2 = new_pred.data
        response = data1.to_numpy()

        pred = data2.to_numpy()

        for n, well in enumerate(well_files):
            if n == 0:
                pass
            else:
                log1 = load_well(file_name=well)

                data, MD = log1.from_csv()

                Por = data['por'].to_numpy()
                Por_df = pd.DataFrame(Por, columns=['Por'])

                data.drop('por', axis=1, inplace=True)

                new_pred = predictor_ext(data)

                for name in win_names:
                    new_pred.roll_and_win_sel_well(data_name=name, win=self.win,
                                                   geo_int=self.geo_int, grid=self.grid2, horizons_list=self.horizons_list, MD=MD, well_path=well_paths_horizons[n])
                new_pred2 = predictor_ext(Por_df)
                new_pred2.remove_outside_window(win=self.win)

                data1 = new_pred2.data
                data2 = new_pred.data

                response2 = data1.to_numpy()

                pred2 = data2.to_numpy()

                response = np.vstack((response, response2))
                pred = np.vstack((pred, pred2))

        # standardize
        pred, self.mean_arr, self.std_arr = new_pred.stand_pred(pred)

        # randomize index to scramble the data
        idx = np.random.rand(*response.shape).argsort(axis=0)
        print(pred)
        response = np.take_along_axis(response, idx, axis=0)
        pred = np.take_along_axis(pred, idx, axis=0)
        print(pred)

        self.response = response
        self.pred = pred

    def load_synthetic_wells_and_predictors_ext(self):
        """
        load the wells from the synthetic sections as predictors and

        Returns
        -------
        None.

        """
        ##################################################################################################################################################

        log1 = load_well(file_name='data_2d_wedge_F3\F03_2_por_eff.xlsx')

        grid_list = [self.grid2, self.grid3]
        self.grid_list = grid_list
        grid_names = ['imp', 'Por']
        self.grid_names = grid_names

        # get the impedance and porosity at the trace == col
        df = log1.from_synthetic(grid_list, grid_names, col=self.wells_loc[0])

        # move the porosity to a different dataframe as it is the target
        Por = df['Por'].to_numpy()
        Por_df = pd.DataFrame(Por, columns=['Por'])

        df.drop('Por', axis=1, inplace=True)
        ##################################################################################################################################################
        name = 'imp'

        new_pred = predictor_ext(df)

        # select values in  window, get window mean and median and add all these as predictors
        new_pred.roll_and_win_sel(data_name=name, win=self.win,
                                  geo_int=self.geo_int, grid=self.grid2, col=self.wells_loc[0],
                                  w_start=self.w_start, w_end_top=self.w_end_top, w_end_base=self.w_end_base,
                                  horizons_list=self.horizons_list, max_TWT=self.max_TWT, min_TWT=self.min_TWT)

        new_pred2 = predictor_ext(Por_df)
        # remove target values outside the window
        new_pred2.remove_outside_window(win=self.win)
        data1 = new_pred2.data
        data2 = new_pred.data

        response = data1.to_numpy()

        pred = data2.to_numpy()
        ##################################################################################################################################################
        if len(self.wells_loc) > 1:
            # for each well location perform the same  predictor extraction as above
            for i in range(len(self.wells_loc)):
                if i == 0:
                    pass
                else:
                    ##################################################################################################################################################
                    log2 = load_well(
                        file_name='data_2d_wedge_F3\F03_2_por_eff.xlsx')

                    df2 = log2.from_synthetic(
                        grid_list, grid_names, col=self.wells_loc[i])

                    Por2 = df2['Por'].to_numpy()
                    Por_df2 = pd.DataFrame(Por2, columns=['Por'])

                    df2.drop('Por', axis=1, inplace=True)
                    ##################################################################################################################################################
                    new_pred = predictor_ext(df2)

                    new_pred.roll_and_win_sel(data_name=name, win=self.win,
                                              geo_int=self.geo_int, grid=self.grid2, col=self.wells_loc[i],
                                              w_start=self.w_start, w_end_top=self.w_end_top, w_end_base=self.w_end_base,
                                              horizons_list=self.horizons_list, max_TWT=self.max_TWT, min_TWT=self.min_TWT)

                    new_pred2 = predictor_ext(Por_df2)
                    new_pred2.remove_outside_window(win=self.win)

                    data1 = new_pred2.data
                    data2 = new_pred.data

                    response2 = data1.to_numpy()

                    pred2 = data2.to_numpy()

                    response = np.vstack((response, response2))
                    pred = np.vstack((pred, pred2))
                    ##################################################################################################################################################
        else:
            pass

        # standardize
        pred, self.mean_arr, self.std_arr = new_pred.stand_pred(pred)

        # randomize index to scramble the data
        idx = np.random.rand(*response.shape).argsort(axis=0)
        print(pred)
        response = np.take_along_axis(response, idx, axis=0)
        pred = np.take_along_axis(pred, idx, axis=0)
        print(pred)

        self.response = response
        self.pred = pred

    def ML_with_cross_val(self, ML_method, n_estimators=np.arange(4, 200, 10), max_depth=np.arange(1, 12, 1),
                          neurons=np.arange(10, 300, 50), layers_list=[1, 3, 5], act_list=['sigmoid'], epochs=100,
                          N=np.arange(1, 100, 1), al_array=np.arange(0, 0.03, 0.0001), cv=10, plot=True):
        """
        Performes machine learning cross-validation and training. 

        Parameters
        ----------
        ML_method : str
            the machine learning method to be used.
        n_estimators : numpy array, optional
            number of decision trees. The default is np.arange(4, 200, 10).
        max_depth : numpy array, optional
            max depth for each decision tree. The default is np.arange(1, 12, 1).
        neurons : numpy array, optional
            range of number of neurans in every layer considered. The default is np.arange(10, 300, 50).
        layers_list : list, optional
            range of number of hidden layers considered. The default is [1,3, 5].
        act_list : list, optional
            activation functions considered. The default is ['sigmoid'].
        epochs : int, optional
            numnber of epochs or cycles of training. The default is 100.
        N : numpy array, optional
            number of neigbors considered. The default is np.arange(1, 100, 1).
        al_array : numpy array, optional
            penalty coefficients considered. The default is np.arange(0, 0.03, 0.0001).
        cv : int, optional
            number of folds. The default is 10.
        plot : bool, optional
            should the cross validation scores the plotted by tuning parameter. The default is True.

        Returns
        -------
        float
            cross-validation MSE (CV score).

        """

        self.ML_method = ML_method
        self.ML = Mlearning(self.pred, self.response, self.grid2)
        uf = usefull_functions()
        self.LOSS = dict()

        # perform cross-validation, all if sections follow a similar logic
        if self.ML_method == 'Random Forest':
            CV_RF = self.ML.RF_cross_val(
                n_estimators=n_estimators, max_depth=max_depth, cv=cv, plot=plot)  # perform cross validation
            # retreve best cross-validation result
            best, best_ = self.handle_CV('CV_RF', CV_RF)

            # fit model according to the best parameters
            self.ML.RF_init(n_estimators=int(best[1]), max_depth=int(best[3]))

        elif self.ML_method == 'Neural Net':
            CV_net = self.ML.n_net_cross_val(
                N=neurons, layers_list=layers_list, act_list=act_list, epochs=epochs, cv=cv, plot=plot)  # perform cross validation
            # retreve best cross-validation result
            best, best_ = self.handle_CV('CV_net', CV_net)

            self.ML.n_net_init(self.pred, self.response, int(best[5]), int(
                best[1]), activation=best[3], epochs=epochs)  # fit model according to the best parameters

        elif self.ML_method == 'lasso':
            CV_lasso = self.ML.lasso_cross_val(
                al_array=al_array, cv=cv, plot=plot)  # perform cross validation

            # retreve best cross-validation result
            best, best_ = self.handle_CV('CV_lasso', CV_lasso)
            # fit model according to the best parameters
            self.ML.lasso_init(float(best[1]))

        else:
            CV_KNN = self.ML.KNN_cross_val(
                N=N, cv=cv, plot=plot)  # perform cross validation
            # retreve best cross-validation result
            best, best_ = self.handle_CV('CV_KNN', CV_KNN)
            # fit model according to the best parameters
            self.ML.KNN_init(int(best[1]))

        self.best_ = best_  # save the best parameters as class object

        return self.neg_mse

    def predict(self, case_type='synthetic'):
        """
        apply trained ML method to case

        Parameters
        ----------
        case_type : str, optional
            if the model is synthetic uses ML.predict_syn_grid, uses ML.predict_syn_grid if not. The default is 'synthetic'.

        Returns
        -------
        None.

        """

        if case_type == 'synthetic':
            if self.ML_method == 'Random Forest':

                pred_map = self.ML.predict_syn_grid(self.grid_list, self.grid_names, self.win, method='RF', geo_int=self.geo_int, 
                                                    w_start=self.w_start, w_end_top=self.w_end_top, w_end_base=self.w_end_base,
                                                    horizons_list=self.horizons_list, max_TWT=self.max_TWT, min_TWT=self.min_TWT,
                                                    mean_arr=self.mean_arr, std_arr=self.std_arr)
            else:

                pred_map = self.ML.predict_syn_grid(self.grid_list, self.grid_names, self.win, method=self.ML_method, geo_int=self.geo_int,
                                                    w_start=self.w_start, w_end_top=self.w_end_top, w_end_base=self.w_end_base,
                                                    horizons_list=self.horizons_list, max_TWT=self.max_TWT, min_TWT=self.min_TWT,
                                                    mean_arr=self.mean_arr, std_arr=self.std_arr)

        else:
            if self.ML_method == 'Random Forest':

                pred_map = self.ML.predict_grid(self.grid_list, self.grid_names, self.win, method='RF', geo_int=self.geo_int,
                                                horizons_list=self.horizons_list, max_TWT=self.max_TWT, min_TWT=self.min_TWT,
                                                mean_arr=self.mean_arr, std_arr=self.std_arr)
            else:

                pred_map = self.ML.predict_grid(self.grid_list, self.grid_names, self.win, method=self.ML_method, geo_int=self.geo_int,
                                                horizons_list=self.horizons_list, max_TWT=self.max_TWT, min_TWT=self.min_TWT,
                                                mean_arr=self.mean_arr, std_arr=self.std_arr)

        self.pred_map = pred_map

    def result_eval(self, vcut=False, hcut=False, ylab='TWT', vminp=0.2, vmaxp=0.4):
        """
        compare the ML prediction against the true porosity, assuming a synthetic model.
        Returns statistical values of the comparison.
        Plots the true porosity, predicted porosity and the absolute difference between the two.

        Parameters
        ----------
        vcut : list, optional
            vertical slice of the array (TWT). The default is False.
        hcut : list, optional
            horizontal slice of the array (traces). The default is False.
        ylab : str, optional
            y label. The default is 'TWT'.
        vminp : float, optional
            minimum porosity value in colormap. The default is 0.2.
        vmaxp : float, optional
            maximum porosity value in colormap. The default is 0.4.

        Returns
        -------
        float
            mean absolute error.
        float
            mean squared error.
        float
            r2 score.
        list
            standard deviation of the absolute error.

        """

        vmin = np.min(self.grid3)
        vmax = np.max(self.grid3)

        # in order to take into account the window size if used
        por_grid = self.grid3[:, self.win:len(self.grid3[0])-self.win]

        pl = plot_results()
        # titles = ['using {}, Wells at {}, window = {}'.format(self.ML_method, str(self.wells_loc),str(self.win)),
        #           'Estimate', 'absolute error with {}, Wells at {}, window = {}'.format(self.ML_method, str(self.wells_loc),str(self.win))]
        titles = ['Prediction on bottom using {}, Wells at {}'.format(self.ML_method, str(self.wells_loc)),
                  'Estimate', 'absolute error with {}, Wells at {}'.format(self.ML_method, str(self.wells_loc))]

        # file name
        best_ = self.best_.replace("/", "")
        file_name = 'results/{}/figures/{}, using {}, wells at {}, window = {}, geo int = {}'.format(
            self.case, self.ML_method, best_, self.wells_loc, str(self.win), self.geo_int)
        file_name = file_name.replace("[", "")
        file_name = file_name.replace("]", "")
        file_name = file_name.replace(":", "")

        pl.grids_comp(por_grid, self.pred_map, titles=titles, vcut=vcut, hcut=hcut, file_name=file_name,
                      extent=self.extent2, ylab=ylab, vmin=vminp, vmax=vmaxp)

        r_squared = r2_score(por_grid.flatten(), self.pred_map.flatten())

        self.LOSS[self.ML_method +
                  ' AE_std'] = np.std(abs(por_grid.flatten()-self.pred_map.flatten()))
        self.LOSS[self.ML_method +
                  ' MAE'] = np.mean(abs(por_grid.flatten()-self.pred_map.flatten()))
        self.LOSS[self.ML_method+' MSE'] = np.mean((por_grid-self.pred_map)**2)
        self.LOSS[self.ML_method+' r2 score'] = r_squared

        res = {'case: ': [self.case],
               'wells location ': [str(self.wells_loc)],
               'method: ': [self.ML_method],
               'window for prediction extraction :': [self.win],
               'Geological interpretation': [self.geo_int],
               'hyperparameter tuning result: ': [best_],
               'Cross validation negative MSE ': [self.neg_mse],
               'validation error MAE: ': [self.LOSS[self.ML_method+' MAE']],
               'validation error MSE: ': [self.LOSS[self.ML_method+' MSE']],
               'abs error std': [self.LOSS[self.ML_method+' AE_std']],
               'r2 score': [self.LOSS[self.ML_method+' r2 score']]
               }
        df = pd.DataFrame(res)
        file_name = file_name.replace("figures", "info")
        df.to_csv(file_name+'.txt', sep='/', index=False, encoding='utf-8')

        return self.LOSS[self.ML_method+' MAE'], self.LOSS[self.ML_method+' MSE'], self.LOSS[self.ML_method+' r2 score'], [self.LOSS[self.ML_method+' AE_std']]

    def result_eval_F3(self, vcut=False, hcut=False, ylab='TWT', vminp=0.2, vmaxp=0.4):
        """
        compare the ML prediction against the true porosity in the F3 case.
        Returns statistical values of the comparison.
        Plots the true porosity, predicted porosity and the absolute difference between the two.

        vcut : list, optional
            vertical slice of the array (TWT). The default is False.
        hcut : list, optional
            horizontal slice of the array (traces). The default is False.
        ylab : str, optional
            y label. The default is 'TWT'.
        vminp : float, optional
            minimum porosity value in colormap. The default is 0.2.
        vmaxp : float, optional
            maximum porosity value in colormap. The default is 0.4.

        Returns
        -------
        res : dict
            dictionary of the parameters used, the cross-validation result and the cross-validation MSE.

        """

        vmin = np.min(self.pred_map)
        vmax = np.max(self.pred_map)

        # in order to take into account the window size if used
        pet_grid = self.grid_pet[:, self.win:len(self.grid_pet[0])-self.win]

        pl = plot_results()
        
        titles = ['Prediction on bottom using {}'.format(self.ML_method),
                  'Estimate', 'absolute difference between solutions']

        # file name
        best_ = self.best_.replace("/", "")
        file_name = 'results/{}/figures/{}, using {}, window = {}, geo int = {}'.format(
            self.case, self.ML_method, best_, str(self.win), self.geo_int)
        file_name = file_name.replace("[", "")
        file_name = file_name.replace("]", "")
        file_name = file_name.replace(":", "")
        pl.grids_comp(pet_grid, self.pred_map, titles=titles, vcut=vcut, hcut=hcut, file_name=file_name,
                      extent=self.extent2, ylab=ylab, vmin=vminp, vmax=vmaxp)

        res = {'case: ': [self.case],
               'method: ': [self.ML_method],
               'window for prediction extraction :': [self.win],
               'Geological interpretation': [self.geo_int],
               'hyperparameter tuning result: ': [best_],
               'Cross validation negative MSE ': [self.neg_mse]
               }
        df = pd.DataFrame(res)
        file_name = file_name.replace("figures", "info")
        df.to_csv(file_name+'.txt', sep='/', index=False, encoding='utf-8')

        return res

    def petrel_solution_eval(self):
        """
        Compare the Sequential Gaussian simulation against the true porosity.
        Returns statistical values of the comparison.
        Plots the true porosity, predicted porosity and the absolute difference between the two.

        Returns
        -------
        MAE : float
            mean absolute error.
        MSE : float
            mean squared error.
        r_2 : float
            r2 score.
        AE_std : float
            standard deviation of the absolute error.

        """
        ext = seismic_ext(self.petrel_solution)
        gridsol = ext.syn_seismic(plot=False, scaling=False)[0]

        if self.case == 'case 2a wedge hetero' or self.case == 'case 2b wedge hetero':
            gridsol = gridsol[:, 0:75]

        pl = plot_results()
        titles = ['True porosity: top, classic solution: below',
                  'Classical solution', 'Difference petrel solution']

        pl.grids_comp(self.grid3, gridsol, titles=titles, file_name='results/{}/figures/{} Petrel_solution'.format(self.case, self.case),
                      extent=self.extent2)

        self.gridsol = gridsol

        AE_std = np.std(abs(self.grid3.flatten() - gridsol.flatten()))
        r_2 = r2_score(self.grid3.flatten(), gridsol.flatten())
        MAE = np.mean(abs(self.grid3.flatten() - gridsol.flatten()))
        MSE = np.mean(((self.grid3.flatten() - gridsol.flatten())**2))

        return MAE, MSE, r_2, AE_std
