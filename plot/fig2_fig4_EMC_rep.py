import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import plot_repeat as pr

from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle

def draw_rep(mode, filepass, savetxtname, savefilename):
    #extract data
    param, subj_list, subj, seed, K_est, gamma_est, K_variance, gamma_variance = pr.read_data(filepass, mode)

    # calculate statistical values
    K_MAP_ave, gamma_MAP_ave, K_MAP_std, gamma_MAP_std, K_MAP_dif_ave, gamma_MAP_dif_ave, K_MAP_dif_std, gamma_MAP_dif_std, K_var_ave, gamma_var_ave, K_var_std, gamma_var_std = pr.calc_stat(param, subj_list, subj, K_est, gamma_est, K_variance, gamma_variance)

    # determine seeds for single EMC simulation
    pr.det_seed(param, subj_list, subj, seed, K_est, gamma_est, K_MAP_dif_ave, gamma_MAP_dif_ave, savetxtname)

    # adjust figure settings
    XLIM, xlabel = pr.change_set(mode)

    # Plot figure
    pr.plot_rep(param, subj_list, K_MAP_ave, gamma_MAP_ave, K_MAP_std, gamma_MAP_std, K_MAP_dif_ave, gamma_MAP_dif_ave, K_MAP_dif_std, gamma_MAP_dif_std, K_var_ave, gamma_var_ave, K_var_std, gamma_var_std, XLIM, xlabel, savefilename)

### Figure 2: OA
mode = 'OA'
filepass = './results/output_files_OA_rep'
savetxtname = "OA_rep_seed.txt"
savefilename = "OA_rep.pdf"

draw_rep(mode, filepass, savetxtname, savefilename)

### Figure 4: kuramoto
mode = 'kuramoto'
filepass = './results/output_files_kuramoto_rep'
savetxtname = "kuramoto_rep_seed.txt"
savefilename = "kuramoto_rep.pdf"

draw_rep(mode, filepass, savetxtname, savefilename)