# SESAMI_GUI
The interface to the SESAMI analysis tools

## Installation instructions
To get the code from this GitHub repo on your computer, open your terminal or command prompt, use `cd` navigate to the folder where you want the SESAMI_GUI to be stored, and run `git clone https://github.com/hjkgrp/SESAMI_GUI.git`.

You will need to install Anaconda and Pip on your computer. These programs are used to download other packages that the SESAMI GUI needs in order to run. Specifically, a Conda environment will be made. The benefit of a Conda environment is that packages in the environment won't conflict with other packages on your computer outside of the Conda environment.

Once you have installed both Anaconda and Pip, proceed to the installation instructions for either Mac or Windows, depending on the operating system of your computer.

### Mac
Using the terminal, navigate to the location where you git cloned the SESAMI_GUI code using `cd`. 
The code that follows will create a conda environment which contains the packages necessary to run the GUI.
If you close and reopen the terminal, you will need to navigate to the SESAMI_GUI folder again and run `conda activate ./.venv` to reactivate the environment.

```
conda create -p ./.venv
conda activate ./.venv
conda install --file conda_package_list_mac.txt
pip install -r pip_package_list_mac.txt
```

### Windows
Using the command prompt, navigate to the location where you git cloned the SESAMI_GUI code using `cd`.
The code that follows will create a conda environment which contains the packages necessary to run the GUI.
If you close and reopen the command prompt, you will need to navigate to the SESAMI_GUI folder again and run `conda activate ./.venv` to reactivate the environment.

```
conda create -p ./.venv
conda activate ./.venv
conda install --file conda_package_list_windows.txt
pip install -r pip_package_list_windows.txt
conda remove --force numpy, scipy
pip install numpy
pip install scipy
```

## Running the GUI
Once you have everything installed, navigate to the location of the SESAMI_GUI folder in the terminal or command prompt. Then, with the .venv environment activate, run the following in the terminal or command prompt: `python trial_GUI_4.py`. Congratulations! The GUI should be running. If not, please do not hesitate to reach out.
