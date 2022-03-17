# Most of the code in this file is from SESAMI_2.0.ipynb, which was released with the paper at https://doi.org/10.1021/acs.jpclett.0c01518

# The structure uploaded to the GUI is called the test_data in this script.
# 

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
from ML import ML # importing the ML class

import pickle

# This function returns the ML prediction of the surface area for the new structure.
def calculation_v2_runner():
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.expand_frame_repr', False)

  plt.style.use(os.path.join('mypaper.mplstyle'))  #This is the matplotlib figure style I have developed based on the Seaborn style. 

  # #Setting the working directory. 
  # os.chdir(os.path.dirname(os.path.realpath('__file__')) )

  ### Below is the implementation section. That is the code to train the ML model.

  ML = ML() #This initiates the class. 

  #This is the description for this particular type of run. All of the output files will have this in their name which can be used to identify them. 
  desc="ML_model" 

  ini_comb = pd.read_csv('ini_Comb.txt', sep='\t', header=0) #The information about Names, true monolayer areas, etc.,
    #is read into the software. 

  isotherm_data_path='isotherm_data' #path to isotherm data.
  output_data_path='Outputs' #Output data path. 

  test_data = pd.DataFrame({'name': ['user_structure']}) # This will hold the features of the new MOF/material that is input at the GUI

  #Actual pressure bins as features. 

  #Here, we are creating the pressure bins which can be used for the ML model. 
  n_bins= 7
  pressure_bins = ML.build_pressure_bins(5,1e5, n_bins)

  #Here, we generate a string for the various columns in the regression which we will input to the formula. 
  feature_list=['c_%d'%i for i in range(len(pressure_bins))]

  #This is the list of features that we want to use. 
  col_list = feature_list + ML.generate_combine_feature_list(feature_list, degree =2)

  #We now proceed to add the features to the dataframe. 
  test_data = ML.add_features(test_data,  pressure_bins, feature_list, col_list, method='actual', set_="Test", isotherm_data_path=isotherm_data_path)
              

  #We output this data to an Excel file. This file helps us understand which fields have NA values. 
  test_data.to_excel(os.path.join(output_data_path,'%s_GUITestData_Na_test.xlsx'%desc), index=False, na_rep='NaN')
  test_data= test_data.dropna(subset=col_list)

  ### Loading the LASSO model
  lasso = pickle.load(open('lasso_model.sav', 'rb'))

  test_data['FittedValues'] = lasso.predict(test_data[col_list])

  # print(f'The surface area prediction for the structure is {test_data.iloc[0]["FittedValues"]}')

  test_prediction = test_data.iloc[0]["FittedValues"]

  return test_prediction

if __name__ == "__main__":
    calculation_v2_runner()