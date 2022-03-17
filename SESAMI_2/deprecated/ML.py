# Most of the code in this file is from SESAMI_2.0.ipynb, which was released with the paper at https://doi.org/10.1021/acs.jpclett.0c01518
# This file contains the class that implements the ML model.

import os
import sys
from pathlib import Path
from IPython.display import Image
import numpy as np
import scipy 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import reduce 

import statsmodels.formula.api as smf
import statsmodels.regression.linear_model as sml

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skm

pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)


plt.style.use(os.path.join('mypaper.mplstyle'))  #This is the matplotlib figure style I have developed based on the Seaborn style. 

#Setting the working directory. 
os.chdir(os.path.dirname(os.path.realpath('__file__')) )

class ML():
    def __init__(self):
        self.N_A = 6.022*10**23 # molecules/mol
        self.A_cs = 0.142*10**-18 # m2/molecule, for argon. #from refs with DOI: 10.1021/acs.langmuir.6b03531 

        # Values by which we will normalize the dataset. Gotten from the training data, in SESAMI_2.0.ipynb
        self.norm_vals = np.array([[1.58953000e-04, 1.00155915e-02, 4.39987176e-02, 3.24635588e-01, 2.05368726e+00, 2.71670605e+00, 2.81671833e+00, 2.52660562e-08, 1.59200831e-06, 6.99372815e-06, 5.16018006e-05, 6.20679622e-04, 2.02395972e-03, 2.28557690e-03, 1.00312072e-04, 4.40673180e-04, 3.25141742e-03, 3.91088782e-02, 1.27529230e-01, 1.44013667e-01, 1.93588715e-03, 1.42835495e-02, 1.71806178e-01, 5.60238764e-01, 6.32655265e-01, 1.05388265e-01, 1.26763694e+00, 4.13360776e+00, 4.66791819e+00, 4.21763134e+00, 6.85987932e+00, 7.11241758e+00, 7.38049176e+00, 7.65219574e+00, 7.93390218e+00,], [4.78708045e+00, 2.68703345e+01, 3.80141609e+01, 4.78106234e+01, 6.17790024e+01, 1.06410565e+02, 1.61478833e+02, 2.29161393e+01, 1.28630453e+02, 1.50296581e+02, 1.60737108e+02, 1.66083055e+02, 1.72479679e+02, 1.80115011e+02, 7.22014876e+02, 8.43628897e+02, 9.02232560e+02, 9.32239866e+02, 9.68144720e+02, 1.01100256e+03, 1.44507643e+03, 1.65793236e+03, 1.73340193e+03, 1.82180829e+03, 1.87702758e+03, 2.28585571e+03, 2.95369262e+03, 3.20344500e+03, 3.25911476e+03, 3.81664514e+03, 4.13936531e+03, 5.18387215e+03, 1.13232083e+04, 1.71830538e+04, 2.60754134e+04,]])

    def initialize_multiple_cols(self, data, col_list, default_val=np.nan):
        """
        In pandas, we cannot initialize multiple columns together with a single value, although we can do that for a single column. So, 
        this function helps us do that.
        data : The dataframe for which we need inititalization.
        col_list : The list of columns which we need intialized.
        default_value : The default value we want in those columns. 
        """
        for col in col_list:
            data[col] = default_val
        return data
        
        
    def pressure_bin_features(self, data, pressure_bins, 
                isotherm_data_path='isotherm_data'):
        """
        This function computes the mean loading for isotherms for the given set of pressure bins and adds it to the given dataframe as columns 
        c_1, c_2, ... c_n. 
        data: Dataframe containing 'Name' and 'TrueSA' columns. 
        pressure_bins: list of tuples giving the start and end points of the pressure ranges we want to use. The interval is half open with equality on
            the greater than side. 
        isotherm_data_path: Path to the location of the isotherm data.
        """
        feature_list=['c_%d'%i for i in range(len(pressure_bins))]
        data = self.initialize_multiple_cols(data, feature_list) #This is the dataframe that will store the feature values.
        print(f'data is ') # TODO remove later
        print(data)

        print(f'pressure_bins are {pressure_bins}') # TODO remove later

        column_names = ['Pressure', 'Loading']
        df = pd.read_table('../SESAMI_1/user_structure/user_isotherm.txt', skiprows=1, sep='\t', names=column_names) # That text file gets made the GUI python script. TODO pass in cwd from GUI

        # print(f'The df is \n{df}') # TODO remove later

        for i, p_bin in enumerate(pressure_bins):
            try:
                val = df.loc[(df['Pressure']>=pressure_bins[i][0] )  & (df['Pressure']< pressure_bins[i][1]), 'Loading'].mean()
                data['c_%d'%i] = val #We are computing the mean values and assigning them to the cell. 
                    # This dataframe is just one row, corresponding to the material described by the isotherm uploaded to the GUI.
            except:
                print ("Feature issue: %s"%name)

        return data
    
    # def p_rel_bin_features(self, data, p_rel_bins, p_sat_method=None, frac_load = 0.99, 
    #                        isotherm_data_path ='isotherm_data' ):
    #     """
    #     This function computes the mean loading for isotherms for the given set of relative pressure bins and adds it to the given dataframe as columns 
    #     cr_1, cr_2, ... cr_n. The difference between prel_bin_features and the pressure_bin_features is that this method helps us 
    #     compute features for the "ML scaled" method mentioned in the paper. 
    #     data: Dataframe containing 'Name' and 'TrueSA' columns. 
    #     p_rel_bins: list of tuples giving the start and end points of the relative pressure ranges we want to use. The interval is half open with equality on
    #         the greater than side. 
    #     p_sat: This is the saturation pressure we want to use to calculate the relative pressure from the actual pressure. If it is float, we use
    #         just that value. If it is none, we compute the pressure at which the loading is is frac_load * qmax. For ex: if the p_sat is None and frac_load = 0.95, we will 
    #         consider the isotherm for the structure, compute the pressure at which loading  = 0.95 * max loading and use that to compute the relative pressure from 
    #         the actual pressure.   
    #     frac_load: Describes the fractional loading to be used as p_sat. Used ONLY when p_sat is None. Ignored otherwise. Function described along with p_sat. 
    #     *** For the current iteration, we do not support a non-None value for p_sat. If this is required, please generate the pressure_bins variable externally and use 
    #         pressure_bin_features. 
    #     isotherm_data_path: Path to the location of the isotherm data.
    #     """
        
    #     feature_list=['cr_%d'%i for i in range(len(p_rel_bins))]
    #     data = self.initialize_multiple_cols(data, feature_list)  
    #     #if p_sat is not None: #This means that there is a float there. In this case, we will just muliply
    #     #    #the p_rel_bins with the p_sat and call the pressure_bin_features function to do the needful. 
               
    #     for name in data['Name'].values:
    #         df= pd.read_table(os.path.join(isotherm_data_path, 'Ar%s.csv'%name),skiprows=2, header=0) #Reads form the location of isotherm data. 
    #         df.loc[df.shape[0]] = [0.0,0.0] #Here, we are adding the point 0,0 to the dataframe in order to start the fitting at (0,0). 
            
    #         if p_sat_method is None:
    #             #next, we compute the p_sat based on frac_load. 
    #             target_load = frac_load * df['Loading'].max() 
    #             #Next, we will interpolate this to compute the desired pressure. We will use linear interpolation. 
    #             lower_index = df[df['Loading']<target_load]['Pressure'].idxmax() ;upper_index = df[df['Loading']>target_load]['Pressure'].idxmin()
    #             p_sat = df.loc[lower_index,'Pressure'] +  (target_load - df.loc[lower_index, 'Loading'])* (df.loc[upper_index,'Pressure'] - df.loc[lower_index, 'Pressure']) / (df.loc[upper_index, 'Loading']-df.loc[lower_index, 'Loading'])
    
    #         elif p_sat_method == "ESW": #we will try to read the ESW index from the isotherm file. 
    #             with open(os.path.join(isotherm_data_path, 'Ar%s.csv'%name)) as file:
    #                 esw_ind =  file.readlines()[0].split()[1]
    
    #             if (esw_ind=="None") or (esw_ind is None): #That means the ESW minimum hasn't been found. In that case, the ESW method cannot be used to scale the isotherm.
    #                 continue
    #             else:
    #                 esw_ind = int(esw_ind)
    #             p_sat = df.loc[esw_ind, 'Pressure']
                                
    #         df['P_rel'] = df['Pressure'] / p_sat 
    #         for i, p_bin in enumerate(p_rel_bins):
    #             try:
    #                 val = df.loc[(df['P_rel']>=p_rel_bins[i][0] )  & (df['P_rel']< p_rel_bins[i][1]), 'Loading'].mean()
    #                 data.loc[data['Name']==name, 'cr_%d'%i] = val #We are computing the mean and assigning the value to the cell. 
    #             except:
    #                 print ("P rel feature issue: No value found for bin %.2e-%.2e for %s"%(p_bin[0], p_bin[1],  name))
    #                 continue
    #     return data
   
    def normalize_df(self, df, col_list, set_="Training", reset_norm_vals="No"):
        """
        This function seeks to normalize the feature columns in the dataframe by the maximum and minimum values of the data. 
        One needs to be careful while applying this function, especially to the test set. It is important to ensure that we are using the same 
        normalization values as the corresponding training set. 
        df: The dataframe whose columns are to be normlized. 
        col_list: The list of columns which need to be normalized.
        set_ : The set (Training or Test) to which the data belongs. 
        reset_norm_vals: If this is set to "Yes", the normalization values will be reset even if they have already been set before. 
        """

        # if set_=="Training":
        #     if type(self.norm_vals)!=np.ndarray or (reset_norm_vals=="Yes") : #If the norm_vals variable hasn't been calculated, it will be done so. 
        #         #print ("Setting normalization values")
        #         self.norm_vals = np.array([  df[col_list].min().values , 
        #                 df[col_list].max().values ] ) 
        # else: #If test set
        #     if type(self.norm_vals)!=np.ndarray: #If the values haven't been set previously. 
        #         raise NameError("Normalization values have not been set. Please normalize training set first.")

        print('debugging') # TODO remove later
        print(f'df[col_list]: {df[col_list]}')
        print(f'self.norm_vals[0,:]: {self.norm_vals[0,:]}')
        print(f'self.norm_vals[1,:]: {self.norm_vals[1,:]}')

        df[col_list] = (df[col_list] - self.norm_vals[0,:] )/ (self.norm_vals[1,:] - self.norm_vals[0,:])

        return df

    def generate_combine_feature_list(self, feature_list, degree =2):
        """
        This function generates the list for 2nd order combined features. 
        feature_list: The list of features which needs to be combined. 
        """
        out_feature_list=[]
        for n1 in np.arange(0, len(feature_list)):
            for n2 in np.arange(n1 , len((feature_list))): 
                el1 = feature_list[n1]
                el2 = feature_list[n2]
                out_feature_list.append(el1+'-'+el2)
        return out_feature_list
    
    def combine_features(self, data, feature_list, degree=2, normalize = "No"):
        """
        This function combines the given features to the required degree. 
        data: The dataframe containing the features. 
        feature_list: The features to be combined.
        degree: The degree to which we want to combine them. In this version, we only support 2nd degree combination. 
        """
        out_feature_list = self.generate_combine_feature_list(feature_list, degree= degree)
        for feature in out_feature_list:
            data[feature] = data[feature.split('-')[0]] * data[feature.split('-')[1]]
        return data
    
    def remove_small_structures(self, df, pressure_criterion=1, selection_criterion = 0.50, 
                        isotherm_data_path = 'isotherm_data' ):
        """
        This function allows us to remove confined pore structures from the dataframe. 
        df: The dataframe containing names and true monolayer areas.
        pressure_criterion: P_check 
        selection_criterion : f_check
        isotherm_data_path : This is the path to the location of the isotherm data. 
        """
                
        datapath= isotherm_data_path

        rejected_list=[]
        selected_list=[]
        for name in df['Name'].values:
            try:
                df_iso = pd.read_csv(os.path.join(datapath, 'Ar%s.csv'%name), skiprows=2, header=0, sep='\t')
                q_max = df_iso['Loading'].max()
                
                if df_iso.loc[df_iso['Pressure']== pressure_criterion, 'Loading'].values[0] / q_max > selection_criterion: #confined pore type. 
                    rejected_list.append(name)
                else:
                    selected_list.append(name)
            except:
                print ("%s : There is a problem finding the maximum loading."%name)
                continue
        rejected_df = df[df['Name'].isin(rejected_list)].copy(deep=True)
        selected_df = df[df['Name'].isin(selected_list)].copy(deep=True)
        
        #we assign a label to the full dataset based on whether or not they were removed
        rejected_df['RemStat']='Removed'
        selected_df['RemStat'] ='Retained'
           
        return [rejected_df, selected_df]
    
    def analyze_crit_performance(self, tr_rej, tr_sel, test_rej, test_sel, output_data="No" , 
                                 output_data_path='Outputs', out_name="", error_criterion=20):
        """
        This function analyzes the performance of the structure removal criterion. 
        tr_rej : Confined pore structures from training set.
        tr_sel: Non confined pore structures from training set. 
        test_rej: Confined pore structures from test set.
        test_sel: Non confined pore structures from test set.
        output_data: Whether or not to write out the statistics of the criterion performance. 
        output_data_path: If output_data=="Yes", the path to where the outputs are to be written. 
                ***Please make sure that this directory exists.
        out_name: If output_data=="Yes", root name of the output data files. 
        error_criterion: The value of max area error that we are interested in analyzing the 
                criterion for. "For example, how many structures with error < 5 were we able to reject with this criterion?"
        """

        #In these lines, "20" in the variable names is assuming that the error criterion is 20. However, the code applies to any value
        #of error_criterion. 
        rej_dev_lt_20 = tr_rej[tr_rej['MaxError'] < error_criterion  ].shape[0] + test_rej[test_rej['MaxError'] < error_criterion  ].shape[0]
        rej_dev_gt_20 = tr_rej[tr_rej['MaxError'] >= error_criterion  ].shape[0] + test_rej[test_rej['MaxError'] >= error_criterion ].shape[0]
        sel_dev_lt_20 = tr_sel[tr_sel['MaxError'] < error_criterion ].shape[0] + test_sel[test_sel['MaxError'] < error_criterion ].shape[0]
        sel_dev_gt_20 = tr_sel[tr_sel['MaxError'] >= error_criterion  ].shape[0] + test_sel[test_sel['MaxError'] >= error_criterion ].shape[0]
        
        tot_rej = pd.concat([tr_rej, test_rej])

        max_error_name = tot_rej.loc[np.abs(tot_rej['MaxError']).idxmax() , 'Name' ]
        max_error_lcd = tot_rej.loc[np.abs(tot_rej['MaxError']).idxmax() , 'LCD' ]
        max_error_true = tot_rej.loc[np.abs(tot_rej['MaxError']).idxmax() , 'TrueSA' ]
        max_error = tot_rej.loc[np.abs(tot_rej['MaxError']).idxmax() , 'MaxError' ]
    
        average_error = np.abs(tot_rej['MaxError']).mean()        
        
        print("Section\tStructures\tDev<%d \tDev>=%d"%(error_criterion,error_criterion )  )
        print("Rejected\t%d\t%d\t%d"%( rej_dev_lt_20+ rej_dev_gt_20  ,rej_dev_lt_20, rej_dev_gt_20))
        print("Selected\t%d\t%d\t%d"%(  sel_dev_lt_20 + sel_dev_gt_20 ,  sel_dev_lt_20, sel_dev_gt_20))
        print ("Low error removal rate: %d"%(rej_dev_lt_20 / (rej_dev_lt_20 + sel_dev_lt_20)*100))
        print ("Max error: Name %s LCD %.1f TrueSA %.0f Error %.2f"%(max_error_name, max_error_lcd, max_error_true, max_error))
        print ("Mean of abslute error: %.2f"%average_error)

        [fig, [ax1, ax2]] = plt.subplots(1,2,figsize=(14,6))
                
        ax1.scatter(tr_rej['MaxLoadArea'], tr_rej['TrueSA'], color=plt.cm.Blues(200)) 
        ax1.scatter(test_rej['MaxLoadArea'], test_rej['TrueSA'], color=plt.cm.Blues(200)) 
        ax1.set_title('Removed'); 
        ax1.scatter(test_rej['MaxLoadArea'], test_rej['TrueSA'], color=plt.cm.Blues(200))
        ax1.set_xlabel('Max load area (m2/g)'); ax1.set_ylabel('True monolayer area (m2/g)')
        ax1.plot( [0, 2500], [0,2500] ) 
    
        ax2.scatter(tr_sel['MaxLoadArea'], tr_sel['TrueSA'], color=plt.cm.Blues(200))
        ax2.scatter(test_sel['MaxLoadArea'], test_sel['TrueSA'], color=plt.cm.Blues(200))
        ax2.set_title('Retained') 
        ax2.set_xlabel('Max load area (m2/g)')
        ax2.set_ylabel('True monolayer area (m2/g)');
        ax2.plot([0,25000], [0,25000] ) 
        ax2.set_xlim((0,10000)); 
        ax2.set_ylim((-400, 10000))
        
        if output_data=="Yes":
            with open(os.path.join(output_data_path, out_name+'.txt'), 'w') as outfile:
                outfile.write( "Section\t#Str\tDev<%d\tDev>%d"%(error_criterion, error_criterion)+"\n" +
                              "Rejected\t%d\t%d\t%d"%( rej_dev_lt_20+ rej_dev_gt_20  ,rej_dev_lt_20, rej_dev_gt_20) + "\n" +
                              "Selected\t%d\t%d\t%d"%(  sel_dev_lt_20 + sel_dev_gt_20 ,  sel_dev_lt_20, sel_dev_gt_20) + "\n" + 
                              "Low error removal rate: %d"%(rej_dev_lt_20 / (rej_dev_lt_20 + sel_dev_lt_20)*100) + "\n" +
                              "Max error: Name %s LCD %.1f TrueSA %.0f Error %.2f"%(max_error_name, max_error_lcd, max_error_true, max_error) + "\n" + 
                              "Mean of abslute error: %.2f"%average_error
                              )
            fig.savefig(os.path.join(output_data_path, out_name+'.png'), bbox_inches='tight')    

    def build_pressure_bins(self, init_val, final_val, n_points ):
        """
        This function creates pressure bins. 
        init_val: The initial value of the pressure bins.
        final_val: The final value of the pressure bins. 
        n_points: The number of points we want to have. 
        """
        p_points = np.logspace(np.log10(init_val) , np.log10(final_val) , n_points) #This is wrt log 10.
        p_points = np.insert( p_points, 0, 0 )

        p_points = np.round(p_points, decimals=0) #Here, we round the numbers to the nearest integer. 

        bins=[]
        for index, point in enumerate(p_points[:-1]):
            bins.append( (p_points[index], p_points[index+1] ) )
        return bins
            
    def build_sym_log_bins( self , n_points, limit=50, base = 10  ):
        """
        This function makes symmetric logarithmic bins symmetric around 1.0. The parameters are: 
        n_points: The number of points above and below 1.0 which are included in the list. The number of bins will be 2*n-1. 
        limit: The upper (1 * limit) and lower (1 / limit) limits between which the points will lie. 
        base : The base with which we want to discretize the log space between the points. 
        """
        
        low_points = np.logspace(np.log(1/limit) / np.log(base), 0 , n_points, base= base)
        high_points = np.logspace(0, np.log(limit) / np.log(base) , n_points, base=base )
        
        bins=[]
        low_points = np.insert( low_points, 0, 0, )

        low_points = np.round(low_points, decimals=4) #Here, we round the numbers to the nearest integer. 
        for index, point in enumerate(low_points[:-1]):
            bins.append( ( low_points[index], low_points[index+1] ) )

        high_points = np.round(high_points, decimals=4) #Here, we round the numbers to the nearest integer.     
        for index, point in enumerate(high_points[:-1]):
            bins.append( ( high_points[index], high_points[index+1] ) )
        return bins

    def add_features(self, tr_data, pressure_bins, feature_list, col_list, method="actual", set_="Training", reset_norm_vals="No", isotherm_data_path='isotherm_data'):
        """
        This function adds the necessary features for the machine learning model. 
        tr_data : The data for which we want to add the features.
        pressure_bins: list of tuples giving the start and end points of the pressure ranges we want to use. The interval is half open with equality on
            the greater than side.
		feature_list: The list of features. Will look something like ['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6']
		col_list: An expanded list of features that includes cross effect terms. Will look something like ['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_0-c_0', 'c_0-c_1', 'c_0-c_2', 'c_0-c_3', 'c_0-c_4', 'c_0-c_5', 'c_0-c_6', 'c_1-c_1', 'c_1-c_2', 'c_1-c_3', 'c_1-c_4', 'c_1-c_5', 'c_1-c_6', 'c_2-c_2', 'c_2-c_3', 'c_2-c_4', 'c_2-c_5', 'c_2-c_6', 'c_3-c_3', 'c_3-c_4', 'c_3-c_5', 'c_3-c_6', 'c_4-c_4', 'c_4-c_5', 'c_4-c_6', 'c_5-c_5', 'c_5-c_6', 'c_6-c_6']
        method : The method we want to use for ML. 
        set_: The set (Training or Test) for which we want to add the features. 
        reset_norm_vals: If this is set to "Yes", the normalization values will be reset even if they have already been set before. 
        The variables set_ and reseet_norm_vals are only required for normalizing the feature set and details are provided in the 
        function "normalize_df". 
        """
        # if method=="actual": #This means we are using ML model to compute true monolayer areas. 
        tr_data = self.pressure_bin_features(tr_data, pressure_bins, isotherm_data_path=isotherm_data_path)
        tr_data = self.combine_features(tr_data, feature_list )
        tr_data = self.normalize_df(tr_data, col_list, set_=set_, reset_norm_vals=reset_norm_vals)

        # elif method=="scaled": #this means that we are using the ML scaled model. 
        #     tr_data = self.p_rel_bin_features(tr_data, pressure_bins, p_sat_method="ESW" , isotherm_data_path=isotherm_data_path)
        #     tr_data =  self.combine_features(tr_data, feature_list )
        #     tr_data = self.normalize_df(tr_data, col_list, set_ = set_, reset_norm_vals=reset_norm_vals)
        return tr_data
    