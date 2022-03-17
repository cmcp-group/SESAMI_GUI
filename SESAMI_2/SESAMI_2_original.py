# Most of the code in this file is from SESAMI_2.0.ipynb, which was released with the paper at https://doi.org/10.1021/acs.jpclett.0c01518

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
from ML_original import ML # importing the ML class

import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)


plt.style.use(os.path.join('mypaper.mplstyle'))  #This is the matplotlib figure style I have developed based on the Seaborn style. 

#Setting the working directory. 
os.chdir(os.path.dirname(os.path.realpath('__file__')) )

### Below is the implementation section. That is the code to train the ML model.

ML = ML() #This initiates the class. 

#This is the description for this particular type of run. All of the output files will have this in their name which can be used to identify them. 
desc="ML_model" 
#desc="ML_scaled"

ini_comb = pd.read_csv('ini_Comb.txt', sep='\t', header=0) #The information about Names, true monolayer areas, etc.,
  #is read into the software. 

isotherm_data_path='isotherm_data' #path to isotherm data.
output_data_path='Outputs' #Output data path. 
    
ini_comb = ini_comb[~ini_comb['Name'].str.match('CNT')]#The CNT structures are excluded as we only want to consider MOFs. 

#The data is split up into training and test sets. 
tr_data = ini_comb[ini_comb['Set']=="Training"]
test_data = ini_comb[ini_comb['Set']=="Test"] # TODO swap this out with a new MOF/material that is input at the GUI
    # TODO for just this one structure, will need to do the classification thing to determine whether to use the ML model
        # TODO just apply the model for new structures, don't do the classifciation
        # TODO have a button to click, that says stuff about the model
            # Like, model may not work at 1Pa if high loading. Model is more intended for sturctures with hierarchical pore structures TODO 

#This is the structure classification criterion. If we do not want any structures to be removed, 
#we can simply set the selection criterion to >= 1.0. 
pressure_criterion=1 # TODO note that this assumes one pressure is 1 Pa 
selection_criterion=0.50

print ("Removal criteria: ")
print ("Pressure criterion: %d"%pressure_criterion)
print ("Selection criterion: %.2f"%selection_criterion)
 
[tr_rej, tr_data] = ML.remove_small_structures(tr_data, pressure_criterion= pressure_criterion,
                                selection_criterion = selection_criterion, isotherm_data_path='isotherm_data') # TODO note that this function uses info in the isotherm_data folder
[test_rej, test_data] = ML.remove_small_structures(test_data, pressure_criterion= pressure_criterion ,
                                selection_criterion = selection_criterion, isotherm_data_path='isotherm_data')

#Next, we analyze the performance of the criterion chosen. 
#NOTE: Comment out this function when selection_criterion >= 1.0 otherwise there will be errors. 
ML.analyze_crit_performance(tr_rej, tr_data, test_rej, test_data, output_data="Yes",
                          output_data_path=output_data_path, out_name="CritStat-%s"%desc, error_criterion=5)

pressure_features='actual' #Actual pressure bins as features. 
#pressure_features='scaled' #Scaled pressure bins as features. 

if pressure_features=="actual": 
    #Here, we are creating the pressure bins which can be used for the ML model. 
    n_bins= 7
    pressure_bins = ML.build_pressure_bins(5,1e5, n_bins)

    #Here, we generate a string for the various columns in the regression which we will input to the formula. 
    feature_list=['c_%d'%i for i in range(len(pressure_bins))]

    #This is the list of features that we want to use. 
    col_list = feature_list + ML.generate_combine_feature_list(feature_list, degree =2)
    print ("Feature used: %s\n"%(col_list[0].split('_')[0])) #This indicates the features that we are using: 
      #pressure (c) or scaled pressure (cr)
    print(f'col_list: {col_list}')

    #We now proceed to add the features to the dataframe. 
    tr_data = ML.add_features(tr_data, pressure_bins, feature_list, col_list, method='actual', set_="Training", isotherm_data_path=isotherm_data_path) # TODO note that this function uses info in the isotherm_data folder
    test_data = ML.add_features(test_data,  pressure_bins, feature_list, col_list, method='actual', set_="Test", isotherm_data_path=isotherm_data_path)
            
elif pressure_features=="scaled":
    #This builds scaled pressure bins which can be used with the ML scaled model. 
    n_points_sym = 4
    limit=150
    p_rel_bins = ML.build_sym_log_bins(n_points_sym , limit=limit, base=10 )

    #Here, we generate a string for the various columns in the regression which we will input to the formula. 
    prel_list=['cr_%d'%i for i in range(len(p_rel_bins))]

    #This is the list of features that we want to use. 
    col_list = prel_list + ML.generate_combine_feature_list(prel_list, degree =2)
    print ("Feature used: %s\n"%(col_list[0].split('_')[0])) #This indicates the features that we are using: 
      #pressure (c) or scaled pressure (cr)

    #We now proceed to add the features to the dataframe. 
    tr_data = ML.add_features(tr_data, p_rel_bins, prel_list, col_list, method='scaled', set_="Training", isotherm_data_path=isotherm_data_path)
    test_data = ML.add_features(test_data, p_rel_bins, prel_list, col_list, method='scaled' , set_="Test", isotherm_data_path=isotherm_data_path)

#We output this data to an Excel file. This file helps us understand which fields have NA values. 
tr_data[['Name', 'TrueSA'] + col_list].to_excel(os.path.join(output_data_path,'%s_TrData_Na_test.xlsx'%desc), index=False, na_rep='NaN')
tr_data= tr_data.dropna(subset=col_list+['TrueSA'])
#tr_data= tr_data.dropna(subset=col_list+['GeoSA']) #Use this when geometric area is the target. 

test_data[['Name', 'TrueSA'] + col_list].to_excel(os.path.join(output_data_path,'%s_TestData_Na_test.xlsx'%desc), index=False, na_rep='NaN')
test_data= test_data.dropna(subset=col_list+['TrueSA'])
#test_data= test_data.dropna(subset=col_list+['GeoSA']) #Use this when geometric area is the target. 



### Now we actually train the scikit-learn model.
#Now, we will train the model on the data and obtain the parameters. 
X = tr_data[col_list]
y = tr_data['TrueSA']
#y=tr_data['GeoSA'] #Use this when geometric area is the target value. 

lasso = Lasso(random_state=0, max_iter=10000, fit_intercept=True, normalize= False) #this is the LASSO object. 
lasso.tol= 0.01
alphas=np.logspace(-2, 7, 100) #These are the values which we test. 

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False, scoring='r2')
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

lasso.alpha= clf.best_params_['alpha'] #We select the best values of alpha.   
lasso.fit(X,y) #We fit this to the data that we have. 

pickle.dump(lasso, open('lasso_model.sav', 'wb')) # Saving the trained LASSO model

tr_data['FittedValues'] = lasso.predict(X)

#Here, we summarize the model parameters. 
coeff_df = pd.DataFrame()
coeff_df['Coefficients']= lasso.coef_
coeff_df['Col'] = col_list
coeff_df.loc[-1, ['Coefficients', 'Col']]=[lasso.intercept_, 'Intercept']
coeff_df.sort_index(inplace=True)
print (coeff_df)

#Summary of the predictions. 
print ("The training R2 = %.3f"%skm.r2_score(tr_data['TrueSA'], tr_data['FittedValues']))
print ("The training RMSE = %.3f"%(scipy.sqrt(skm.mean_squared_error(tr_data['TrueSA'], tr_data['FittedValues']))))
print ("The training MAE = %.3f"%(scipy.sqrt(skm.mean_absolute_error(tr_data['TrueSA'], tr_data['FittedValues']))))

test_data['FittedValues'] = lasso.predict(test_data[col_list]) 

print ("The test R2 = %.3f"%skm.r2_score(test_data['TrueSA'], test_data['FittedValues']))
print ("The test RMSE = %.3f"%(scipy.sqrt(skm.mean_squared_error(test_data['TrueSA'], test_data['FittedValues']))))
print ("The test MAE = %.3f"%(scipy.sqrt(skm.mean_absolute_error(test_data['TrueSA'], test_data['FittedValues']))))

#We combine the data that we have in our final analysis. 
#comb_data = pd.concat([tr_data, test_data])
#Here, we make a comb_data to get statistics for the full fitting. We recombine the removed structures back to this as well. 
tr_rej['FittedValues'] = tr_rej['MaxLoadArea']
test_rej['FittedValues'] = test_rej['MaxLoadArea']
comb_data = pd.concat([tr_data, test_data, tr_rej, test_rej], sort=False, join='outer')

comb_data['FitError']= (comb_data['FittedValues'] - comb_data['TrueSA']) / comb_data['TrueSA'] * 100
comb_data['BETError']= (comb_data['BETSA'] - comb_data['TrueSA']) / comb_data['TrueSA'] * 100
print( comb_data[np.abs(comb_data['FitError'])>30][['Name', 'TrueSA', 'LCD', 'FitError', 'BETError'] ])
print ("Number of structures: %d"%comb_data.shape[0])

#We now write out these outputs
#This only corresponds to the ML-base fit. 
coeff_df.to_csv(os.path.join(output_data_path, 'Coeff-%s.txt'%desc ), index=False, sep='\t', float_format="%.0f")
with open(os.path.join(output_data_path, 'R2-%s.txt'%desc), 'w') as outfile:
    outfile.write(   
    "# structures in final set = %d"%comb_data.shape[0] + "\n" + 
    "The training R2 = %.3f"%skm.r2_score(tr_data['TrueSA'], tr_data['FittedValues']) + '\n' + 
    "The training RMSE = %.3f"%(scipy.sqrt(skm.mean_squared_error(tr_data['TrueSA'], tr_data['FittedValues']))) + '\n' + 
    "The training MAE = %.3f"%(scipy.sqrt(skm.mean_absolute_error(tr_data['TrueSA'], tr_data['FittedValues']))) + '\n' +
    "The test R2 = %.3f"%skm.r2_score(test_data['TrueSA'], test_data['FittedValues']) + '\n' + 
    "The test RMSE = %.3f"%(scipy.sqrt(skm.mean_squared_error(test_data['TrueSA'], test_data['FittedValues']))) + '\n' +
    "The test MAE = %.3f"%(scipy.sqrt(skm.mean_absolute_error(test_data['TrueSA'], test_data['FittedValues'])))  )

#Severe outliers
comb_data[np.abs(comb_data['FitError'])>30][['Name', 'TrueSA', 'FitError', 'BETError'] ].to_csv(os.path.join(output_data_path,'SevereOut-%s.txt'%desc), 
                                                                        index=False, sep='\t', float_format="%.0f")
#status of CNTs. 
comb_data[comb_data['Name'].str.match('CNT')][['Name', 'TrueSA', 'FittedValues','FitError', 'BETSA','BETError']].to_csv(os.path.join(output_data_path,'CNT_data-%s.txt'%desc), 
                                                                        index=False, sep='\t', float_format="%.0f")
#Full comb data. 
comb_data.to_csv(os.path.join(output_data_path, 'RawOutData-%s.txt'%desc), index=False, sep='\t',  na_rep='NaN')



### Summarizing model training next.
#Here, we make a figure to show cross validation.  
fig= plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_xlim((1e-2, 1e4))
ax.set_ylim((0,1.1))
ax.set_xlabel('$\mathrm{\lambda}$')
ax.set_ylabel('$\mathrm{R^2\ score}$')
ax.plot(alphas, scores, linewidth=2)
ax.fill_between( alphas, scores - scores_std, scores + scores_std, color = plt.cm.Purples(100), alpha=0.4)
ax.vlines(clf.best_params_['alpha'], ax.get_ylim()[0], ax.get_ylim()[1], linestyles='--', color=plt.cm.Greys(150), linewidths=2)
ax.text(0.45, 0.3, '$\mathrm{\lambda^* = %.2f}$'%clf.best_params_['alpha'] + "\n"+ "$\mathrm{R^2=%.3f}$" %clf.best_score_,
horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,  fontdict={'size':18, 'weight':'bold'})
fig.savefig( os.path.join(output_data_path,'Fit-test_%s.png'%desc), format='png', bbox_inches='tight')


#This is the figure to summarize the ML performance. 
[fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]]] = plt.subplots(3, 2, figsize=(14,18))

ax1.set_xlabel('Actual true monolayer area ($\mathrm{m^2/g}$)')
ax1.set_ylabel('Predicted true monolayer area ($\mathrm{m^2/g}$)')
ax1.scatter(comb_data[comb_data['TrueSA']<1500]['TrueSA'], comb_data[comb_data['TrueSA']<1500]['FittedValues'], 
            color=plt.cm.Set1(1), label='Predicted')
ax1.scatter(comb_data[comb_data['TrueSA']<1500]['TrueSA'], comb_data[comb_data['TrueSA']<1500]['BETSA'], 
            color= plt.cm.Set1(0), alpha=0.2, label='BET')
ax1.set_xlim((0, 1600))
ax1.set_ylim((0,1800))
ax1.plot([0, 10000], [0, 10000], linestyle='--', linewidth=2, color=plt.cm.Greys(100))
ax1.text(0.01, 0.99, 'Region I', transform = ax1.transAxes , horizontalalignment='left', verticalalignment='top')
ax1.legend(loc='lower right')

ax2.set_xlabel('Actual true monolayer area ($\mathrm{m^2/g}$)')
ax2.set_ylabel('Predicted true monolayer area ($\mathrm{m^2/g}$)')
ax2.scatter(comb_data[ (comb_data['TrueSA']>1500) & (comb_data['TrueSA']<3500) ]['TrueSA'], comb_data[(comb_data['TrueSA']>1500) & (comb_data['TrueSA']<3500)]['FittedValues'], color=plt.cm.Set1(1), label='Predicted')
ax2.scatter(comb_data[ (comb_data['TrueSA']>1500) & (comb_data['TrueSA']<3500) ]['TrueSA'], comb_data[(comb_data['TrueSA']>1500) & (comb_data['TrueSA']<3500)]['BETSA'], color= plt.cm.Set1(0), alpha=0.2, label='BET')
ax2.set_xlim((1400, 3600))
ax2.set_ylim((1000,4000))
ax2.plot([0, 10000], [0, 10000], linestyle='--', linewidth=2, color=plt.cm.Greys(100))
ax2.text(0.01, 0.99, 'Region II', transform = ax2.transAxes , horizontalalignment='left', verticalalignment='top')
ax2.legend(loc='lower right')

ax3.set_xlabel('Actual true monolayer area ($\mathrm{m^2/g}$)')
ax3.set_ylabel('Predicted true monolayer area ($\mathrm{m^2/g}$)')
ax3.scatter(comb_data[comb_data['TrueSA']>3500]['TrueSA'], comb_data[comb_data['TrueSA']>3500]['FittedValues'], color=plt.cm.Set1(1), label='Predicted')
ax3.scatter(comb_data[comb_data['TrueSA']>3500]['TrueSA'], comb_data[comb_data['TrueSA']>3500]['BETSA'], color= plt.cm.Set1(0), alpha=0.2, label='BET')
ax3.set_xlim((3000, 10000))
ax3.set_ylim((1500,10000))
ax3.plot([0, 10000], [0, 10000], linestyle='--', linewidth=2, color=plt.cm.Greys(100))
ax3.text(0.01, 0.99, 'Region III', transform = ax3.transAxes , horizontalalignment='left', verticalalignment='top')
ax3.legend(loc='lower right')

ax4.set_xlabel('Actual true monolayer area ($\mathrm{m^2/g}$)')
ax4.set_ylabel('Predicted true monolayer area ($\mathrm{m^2/g}$)')
ax4.scatter(comb_data[comb_data['Name'].str.match('CNT')]['TrueSA'], comb_data[comb_data['Name'].str.match('CNT')]['FittedValues'], color=plt.cm.Set1(1), label='Predicted')
ax4.scatter(comb_data[comb_data['Name'].str.match('CNT')]['TrueSA'], comb_data[comb_data['Name'].str.match('CNT')]['BETSA'], color= plt.cm.Set1(0), alpha=0.2, label='BET')
ax4.set_xlim((0, 2000))
ax4.set_ylim((0,2000))
ax4.plot([0, 10000], [0, 10000], linestyle='--', linewidth=2, color=plt.cm.Greys(100))
ax4.text(0.01, 0.99, 'CNTs', transform = ax4.transAxes , horizontalalignment='left', verticalalignment='top')
ax4.legend(loc='lower right')

ax5.set_xlabel('Actual true monolayer area ($\mathrm{m^2/g}$)')
ax5.set_ylabel('Predicted true monolayer area ($\mathrm{m^2/g}$)')
ax5.scatter(tr_data['TrueSA'], tr_data['FittedValues'], color=plt.cm.Set1(1), label='Predicted')
ax5.scatter(tr_data['TrueSA'], tr_data['BETSA'], color= plt.cm.Set1(0), alpha=0.2, label='BET')
ax5.set_xlim((0,10000))
ax5.set_ylim((0,10000))
ax5.plot([0, 10000], [0, 10000], linestyle='--', linewidth=2, color=plt.cm.Greys(100))
ax5.text(0.01, 0.99, 'Training data', transform = ax5.transAxes , horizontalalignment='left', verticalalignment='top')
ax5.legend(loc='lower right')

ax6.set_xlabel('Actual true monolayer area ($\mathrm{m^2/g}$)')
ax6.set_ylabel('Predicted true monolayer area ($\mathrm{m^2/g}$)')
ax6.scatter(test_data['TrueSA'], test_data['FittedValues'], color=plt.cm.Set1(1), label='Predicted')
ax6.scatter(test_data['TrueSA'], test_data['BETSA'], color= plt.cm.Set1(0), alpha=0.2, label='BET')
ax6.set_xlim((0,10000))
ax6.set_ylim((0,10000))
ax6.plot([0, 10000], [0, 10000], linestyle='--', linewidth=2, color=plt.cm.Greys(100))
ax6.text(0.01, 0.99, 'Test data', transform = ax6.transAxes , horizontalalignment='left', verticalalignment='top')
ax6.legend(loc='lower right')

fig.savefig(os.path.join(output_data_path,'FullFig-%s.png'%desc), format='png', dpi=300, bbox_inches='tight')
print(f'There should be a figure at {output_data_path} FullFig-{desc}.png')

print('Made it to the end!') # TODO remove at end
