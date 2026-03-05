#!/usr/bin/env python

### Script to plot outputs of biomechanics analysis
# Python version 3.6 
# Run in the directory the data was previously analysed in
# Required packages: os, pandas, numpy, sys, matplotlib, scipy, re


## Load packages

import pandas as pd
import numpy as np
import os
import sys 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import re

# If any packages aren't installed use package manager (preferably anaconda) to install, e.g.
# conda install -c conda-forge terminaltables

## Read in data
# Get current working directory 
# Assign names of sub-directories for analysis to variable
dir = os.getcwd()
names = [ f.name for f in os.scandir(dir) if f.is_dir() ]


##### PLOT DATA #####
for name in names:
    if name.endswith('Data'):
        print('Plotting results for {}...\n'.format(name))

        # Extract failure and precon data
        precon_df = pd.read_csv("{}/{}/precon_{}.csv".format(dir, name, name), header=0)
        failure_df = pd.read_csv("{}/{}/failure_{}.csv".format(dir, name, name), header=0)

        # Precon plot 1 

        print('Preconditioning plot 1\n')
        x = precon_df['Time_S']
        y = precon_df['Load_correction']
        yhat = scipy.signal.savgol_filter(y, 1001, 3) # window size 1001, polynomial order 3

        plt.figure()
        plt.plot(x, y, 'k-', label='Raw data', color='black', alpha=0.2) # set transparency with 'alpha' parameter
        plt.plot(x, yhat, 'k-', label='Smoothed data', color='red')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Force (N)')
        plt.title('Preconditioning cycle load',fontsize=12)
        plt.legend()
        plt.savefig('{}/{}/precon_cycle_load_{}.png'.format(dir, name, name), bbox_inches='tight',dpi=300)
        plt.close()

        # Precon plot 2

        print('Preconditioning plot 2\n')
        x = precon_df['Displacement_correction']
        y = precon_df['Force_N']
        yhat = scipy.signal.savgol_filter(y, 301, 3) # window size 2001, polynomial order 3

        plt.figure()
        plt.plot(x, y, 'k-', label='Raw data', color='black', alpha=0.2) # set transparency with 'alpha' parameter
        plt.plot(x, yhat, 'k-', label='Smoothed data', color='red')
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Force (N)')
        plt.title('Preconditioning load vs displacement',fontsize=12)
        plt.legend()
        plt.savefig('{}/{}/precon_load_vs_displacement_{}.png'.format(dir, name, name), bbox_inches='tight',dpi=300)
        plt.close()

        # Failure plot 1

        print('Failure plot 1\n')
        x = failure_df['Displacement_correction']
        y = failure_df['Load_correction']
        yhat = scipy.signal.savgol_filter(y, 101, 3) # window size 201, polynomial order 3

        plt.figure()
        plt.plot(x, y, 'k-', label='Raw data', color='black', alpha=0.2) # set transparency with 'alpha' parameter
        plt.plot(x, yhat, 'k-', label='Smoothed data', color='red')
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Force (N)')
        plt.title('Failure Force',fontsize=12)
        plt.legend()
        plt.savefig('{}/{}/failure_force_{}.png'.format(dir, name, name), bbox_inches='tight',dpi=300)
        plt.close()

        # Failure plot 2

        print('Failure plot 2\n')
        x = failure_df['Strain_%']
        y = failure_df['Stress_Mpas']
        yhat = scipy.signal.savgol_filter(y, 101, 3) # window size 201, polynomial order 3

        plt.figure()
        plt.plot(x, y, 'k-', label='Raw data', color='black', alpha=0.2) # set transparency with 'alpha' parameter
        plt.plot(x, yhat, 'k-', label='Smoothed data', color='red')
        plt.xlabel('Strain (%)')
        plt.ylabel('Stress (MPa)')
        plt.title('Stress vs Strain',fontsize=12)
        plt.legend()
        plt.savefig('{}/{}/failure_stress-vs-strain_{}.png'.format(dir, name, name), bbox_inches='tight',dpi=300)
        plt.close()
    
    else:
        print('{} is not an analysis folder...\n'.format(name))



