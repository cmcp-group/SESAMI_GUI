# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:27:22 2019

@author: datar.10
"""

import os 
import sys
import pandas as pd

#We will read in the isotherm file with an option to specify the ESW minima and the consistency 1 limit. 
with open('input.txt', 'r') as in_file:
    in_file = [item.strip() for item in in_file.readlines()]
    name= in_file[1]
    #isopath
    if in_file[3] == 'Default':
        isopath = os.path.join(os.curdir, '%s.txt'%name)
    else: 
        isopath = in_file[3]
    #output path
    if in_file[5] == 'Default':
        output_path = os.curdir
    else:
        output_path = in_file[5]
    #ESW minima info
    if in_file[7] == 'Default':
        eswminimamanual='No'
    else:
        eswminimamanual='Yes'
    #Con1 max info
    if in_file[9] == 'Default':
        con1manual='No'
    else:
        con1manual='Yes'
    #Additional parameters. 
    #R2 min
    par_list = in_file[11].split('\t')
    if par_list[0] == 'Default':
        R2min = 0.998
    else:
        R2min = float(par_list[0])
    #R2 cutoff
    if par_list[1] == 'Default':
        R2cutoff = 0.9995
        
    else:
        R2cutoff = float(par_list[1])
    #minimum length of the region. 
    if par_list[2] == 'Default':
        minlinelength = 4
    else:
        minlinelength = float(par_list[2])
    #path to executable
    if in_file[13] =='Default':
        py_path = os.path.join(os.curdir , '..')
    else: 
        py_path = in_file[14]
    
sys.path.append(py_path)

from betan import BETAn
b= BETAn()

#extradata
b.R2min = R2min
b.R2cutoff = R2cutoff
b.minlinelength = minlinelength


'''
name='CIGXIA'
isopath = os.path.join(home, 'priyadata', 'MOFAnalysis', 'Summary', 'ArCIGXIA.csv')
output_path = os.path.join(os.curdir, name)

eswminimamanual ='No'
con1manual='No'
'''

data = pd.read_table(isopath, skiprows=3, sep='\t', names=['Pressure', 'Loading'])

if eswminimamanual=="Yes":
    b.eswminimamanual="Yes"
    with open(isopath, 'r') as isofile:
        b.eswminima = int(isofile.readlines()[0].strip().split('\t')[-1])
if con1manual=="Yes":
    b.con1limitmanual="Yes"
    with open(isopath, 'r') as isofile:
        b.con1limit = int(isofile.readlines()[1].strip().split('\t')[-1])

data = b.prepdata(data)

if not os.path.isdir(output_path):
    os.mkdir(output_path, mode = 711) #We make the output path

b.generatesummary(data, name=name, filepath=output_path, filename='Output.txt', eswpoints=3, sumpath=output_path, df_path=output_path)


