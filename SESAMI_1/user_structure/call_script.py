# -*- coding: utf-8 -*-

import os 
import sys
import pandas as pd

def calculation_runner(cwd, gas, temperature, p0, plotting_information):
    # We will read in the isotherm file with an option to specify the ESW minima and the consistency 1 limit. 

    # Takes five parameters. The current working directory, the gas used, the temperature, the saturation pressure, and plotting information.
        # Plotting information is a dictionary. The other parameters are strings and floats.

    with open('input.txt', 'r') as in_file:
        in_file = [item.strip() for item in in_file.readlines()]
        name= in_file[1] # user_isotherm
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

    # TODO calculate p0 in the case of CSV, for CO2 and krypton
        # go off of selected gas and temperature
    if p0 == None: # CSV case
        p0 = 1e5 # TODO adjust later

    # The R^2 information from the GUI supersedes the input.txt content R^2   
    b = BETAn(gas, temperature, minlinelength, plotting_information)

    # print('Data check')
    # from IPython.display import display
    # display(data)
    # print(f'dataframe shape: {data.shape}')

    # Commented the code below out, since it will never be the case for the GUI with how the input file is. 
    # Also our generated isotherm text files don't have the first two lines like that. See function CSV_convert in the GUI python script.
    
    # if eswminimamanual=="Yes":
    #     b.eswminimamanual="Yes"
    #     with open(isopath, 'r') as isofile:
    #         b.eswminima = int(isofile.readlines()[0].strip().split('\t')[-1])
    # if con1manual=="Yes":
    #     b.con1limitmanual="Yes"
    #     with open(isopath, 'r') as isofile:
    #         b.con1limit = int(isofile.readlines()[1].strip().split('\t')[-1])

    column_names = ['Pressure', 'Loading']
    # Adjustment from SI code: skipping 1 row rather than three
    data = pd.read_table(isopath, skiprows=1, sep='\t', names=column_names)        
        # Adjustment from SI code: skipping 1 row rather than three
    data = b.prepdata(data, p0=p0)

    if not os.path.isdir(output_path):
        os.mkdir(output_path, mode = 711) #We make the output path

    BET_dict, BET_ESW_dict = b.generatesummary(data, plotting_information, name=name, filepath=output_path, filename='Output.txt', eswpoints=3, sumpath=cwd, df_path=output_path)
        # Change from SI code: path for figures is now the current working directory of the GUI code.

    #     # plotting_information = {'dpi': my_dpi, 'font size': my_font_size, 'font type': my_font_type, 'legend': legend_bool,
    #        'R2 cutoff': R2cutoff, 'R2 min': R2min} # TODO

    return BET_dict, BET_ESW_dict

if __name__ == "__main__":
    my_cwd = os.getcwd()
    calculation_runner(my_cwd)