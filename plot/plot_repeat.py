'''
Functions to plot figures from repeated EMC simulation data. 
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle

### General settings for figure
labelsize = 22
labelsize2 = 16 # ylabel for variance
legendsize = 20
panellabelsize = 27
config = {
    'font.size': 14,
    'lines.markersize': 3.0,
    'lines.markeredgewidth': 0.7,
    'lines.markerfacecolor': 'w',
    # 'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'cm',
}
plt.rcParams.update(config)
myblue = '#4DC4FF'
mygreen = '#03AF7A'


### Functions
# extract data
def read_data(filepass, mode):
    # true parameters
    param = pd.read_csv(filepass + "/true_parameters.csv")

    # results of EMC
    if mode=="OA":
        subj_list = np.loadtxt(filepass + "/sigma_list.txt", usecols = [0], unpack=True)
        subj, seed, K_est, gamma_est, K_variance, gamma_variance = np.loadtxt(filepass + "/result_rep.txt", usecols = [0, 1, 2, 3, 4, 5], unpack=True)
    elif mode=="kuramoto": 
        subj_list = np.loadtxt(filepass + "/oscillator_num_list.txt", usecols = [0], unpack=True)
        subj, seed, K_est, gamma_est, K_variance, gamma_variance = np.loadtxt(filepass + "/result_rep_kuramoto.txt", usecols = [0, 1, 2, 3, 4, 5], unpack=True)

    return param, subj_list, subj, seed, K_est, gamma_est, K_variance, gamma_variance

# calculate statistical values
def calc_stat(param, subj_list, subj, K_est, gamma_est, K_variance, gamma_variance):
    K_true = param['K_true'][0]
    gamma_true = param['gamma_true'][0]

    # prepare empty arrays
    K_MAP_ave, gamma_MAP_ave, K_MAP_std, gamma_MAP_std, K_MAP_dif_ave, gamma_MAP_dif_ave, K_MAP_dif_std, gamma_MAP_dif_std, K_var_ave, gamma_var_ave, K_var_std, gamma_var_std = [np.array([]) for _ in range(12)]

    # calculate statistical values
    for i in subj_list:
        K_MAP_ave = np.append(K_MAP_ave, np.mean(K_est[np.where(subj == i)]))
        K_MAP_std = np.append(K_MAP_std, np.std(K_est[np.where(subj == i)]))
        
        gamma_MAP_ave = np.append(gamma_MAP_ave, np.mean(gamma_est[np.where(subj == i)]))
        gamma_MAP_std = np.append(gamma_MAP_std, np.std(gamma_est[np.where(subj == i)]))
        
        K_MAP_dif_ave = np.append(K_MAP_dif_ave, np.mean(np.abs(K_est[np.where(subj == i)] - K_true)))
        K_MAP_dif_std = np.append(K_MAP_dif_std, np.std(np.abs(K_est[np.where(subj == i)] - K_true)))
        
        gamma_MAP_dif_ave = np.append(gamma_MAP_dif_ave, np.mean(np.abs(gamma_est[np.where(subj == i)] - gamma_true)))
        gamma_MAP_dif_std = np.append(gamma_MAP_dif_std, np.std(np.abs(gamma_est[np.where(subj == i)] - gamma_true)))
        
        K_var_ave = np.append(K_var_ave, np.mean(K_variance[np.where(subj == i)]))
        K_var_std = np.append(K_var_std, np.std(K_variance[np.where(subj == i)]))
        
        gamma_var_ave = np.append(gamma_var_ave, np.mean(gamma_variance[np.where(subj == i)]))
        gamma_var_std = np.append(gamma_var_std, np.std(gamma_variance[np.where(subj == i)]))

    return K_MAP_ave, gamma_MAP_ave, K_MAP_std, gamma_MAP_std, K_MAP_dif_ave, gamma_MAP_dif_ave, K_MAP_dif_std, gamma_MAP_dif_std, K_var_ave, gamma_var_ave, K_var_std, gamma_var_std

# Determine and record which seed to be used for single EMC simulation
def det_seed(param, subj_list, subj, seed, K_est, gamma_est, K_MAP_dif_ave, gamma_MAP_dif_ave, savetxtname):
    K_true = param['K_true'][0]
    gamma_true = param['gamma_true'][0]

    with open(savetxtname, 'w') as f:
        for i in subj_list:
            K_est_temp = K_est[np.where(subj == i)]
            gamma_est_temp = gamma_est[np.where(subj == i)]
            seed_temp = seed[np.where(subj == i)]

            # determine the seed with which the result is closest to the mean value of |K_true - K_MAP|
            T = (np.abs(K_est_temp - K_true) - K_MAP_dif_ave[np.where(subj_list == i)])**2 + (np.abs(gamma_est_temp - gamma_true) - gamma_MAP_dif_ave[np.where(subj_list == i)])**2
            index = np.argmin(T)

            # record
            f.write("subj={}, seed={}, K_infer={}, gamma_infer={} \n".format(i, seed_temp[index], K_est_temp[index], gamma_est_temp[index]))

# Change figure settings between OA and Kuramoto
def change_set(mode):
    if mode=='OA':
        XLIM = [5e-4, 0.5]
        xlabel = r'$\sigma$'
    elif mode=='kuramoto':
        XLIM = [6, 2e5]
        xlabel = r'$N$'
    return XLIM, xlabel

# Plot figure
def plot_rep(param, subj_list, K_MAP_ave, gamma_MAP_ave, K_MAP_std, gamma_MAP_std, K_MAP_dif_ave, gamma_MAP_dif_ave, K_MAP_dif_std, gamma_MAP_dif_std, K_var_ave, gamma_var_ave, K_var_std, gamma_var_std, XLIM, xlabel, savefilename):
    K_true = param['K_true'][0]
    gamma_true = param['gamma_true'][0]

    # draw figure
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=0.2)

    ax1 = fig.add_subplot(321)
    ax1.errorbar(subj_list, K_MAP_ave, yerr=K_MAP_std, color = myblue, marker=".", markersize=20, label = r"$\hat{K}$", capthick=1, capsize=5, lw=3)
    ax1.hlines(K_true, 0, 1e6, color = 'black', linestyle = 'dotted', lw = 2, label = r"$K_{\rm true}$")
    ax1.set_xscale('log')
    ax1.set_xlim(XLIM)
    ax1.set_ylabel('MAP', fontsize = labelsize)
    ax1.legend(fontsize=legendsize)
    ax1.text(-0.02, 1.12, '(a)', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, fontsize=panellabelsize)

    ax2 = fig.add_subplot(322)
    ax2.errorbar(subj_list, gamma_MAP_ave, yerr=gamma_MAP_std, color = mygreen, marker=".", markersize=20, label = r"$\hat{\gamma}$", capthick=1, capsize=5, lw=3)
    ax2.hlines(gamma_true, 0, 1e6, color = 'black', linestyle = 'dotted', lw = 2, label = r"$\gamma_{\rm true}$")
    ax2.set_xscale('log')
    ax2.set_xlim(XLIM)
    # ax2.set_ylabel(r'$\hat{\gamma}$', fontsize = labelsize)
    ax2.legend(fontsize=legendsize)
    ax2.text(-0.02, 1.12, '(b)', horizontalalignment='right', verticalalignment='top', transform=ax2.transAxes, fontsize=panellabelsize)

    ax3 = fig.add_subplot(323)
    ax3.errorbar(subj_list, K_MAP_dif_ave, yerr=K_MAP_dif_std, color = myblue, marker=".", markersize=20, capthick=1, capsize=5, lw=3, label = r"$|K_{\rm true}-\hat{K}|$")
    ax3.hlines(0, 0, 1e6, color = 'black', linestyle = 'solid', linewidth=0.5)
    ax3.set_xscale('log')
    ax3.set_xlim(XLIM)
    ax3.set_ylabel('Difference', fontsize = labelsize)
    ax3.legend(fontsize=legendsize)
    ax3.text(-0.02, 1.12, '(c)', horizontalalignment='right', verticalalignment='top', transform=ax3.transAxes, fontsize=panellabelsize)

    ax4 = fig.add_subplot(324)
    ax4.errorbar(subj_list, gamma_MAP_dif_ave, yerr=gamma_MAP_dif_std, color = mygreen, marker=".", markersize=20, capthick=1, capsize=5, lw=3, label = r"$|\gamma_{\rm true} - \hat{\gamma}|$")
    ax4.hlines(0, 0, 1e6, color = 'black', linestyle = 'solid', linewidth=0.5)
    ax4.set_xscale('log')
    ax4.set_xlim(XLIM)
    ax4.legend(fontsize=legendsize)
    # ax4.set_ylabel(r'$|\gamma_{\rm true} - \hat{\gamma}|$', fontsize = labelsize)
    ax4.text(-0.02, 1.12, '(d)', horizontalalignment='right', verticalalignment='top', transform=ax4.transAxes, fontsize=panellabelsize)

    ax5 = fig.add_subplot(325)
    ax5.errorbar(subj_list, K_var_ave, yerr=K_var_std, color = myblue, marker=".", markersize=20, label = r'${\rm Var}\: [\hat{p}(K|D,b_{\hat{l}})]$', capthick=1, capsize=5, lw=3)
    ax5.set_xscale('log')
    ax5.set_xlim(XLIM)
    ax5.hlines(0, 0, 1e6, color = 'black', linestyle = 'solid', linewidth=0.5)
    #ax1.set_xlabel(xlabel, fontsize = labelsize)
    ax5.legend(fontsize = legendsize)
    ax5.ticklabel_format(axis="y", scilimits=(-2, 1), useMathText=True)
    ax5.text(-0.02, 1.12, '(e)', horizontalalignment='right', verticalalignment='top', transform=ax5.transAxes, fontsize=panellabelsize)
    ax5.set_xlabel(xlabel, fontsize = labelsize)
    ax5.set_ylabel('Variance of estimated\n posterior distribution', fontsize = labelsize, x = -0.05)

    ax6 = fig.add_subplot(326)
    ax6.errorbar(subj_list, gamma_var_ave, yerr=gamma_var_std, color = mygreen, marker=".", markersize=20, label = r'${\rm Var}\: [\hat{p}(\gamma|D,b_{\hat{l}})]$', capthick=1, capsize=5, lw=3)
    ax6.set_xscale('log')
    ax6.set_xlim(XLIM)
    ax6.hlines(0, 0, 1e6, color = 'black', linestyle = 'solid', linewidth=0.5)
    ax6.set_xlabel(xlabel, fontsize = labelsize)
    ax6.legend(fontsize = legendsize)
    ax6.ticklabel_format(axis="y", scilimits=(-2, 1), useMathText=True)
    ax6.text(-0.02, 1.12, '(f)', horizontalalignment='right', verticalalignment='top', transform=ax6.transAxes, fontsize=panellabelsize)
    
    fig.align_labels()
    fig.savefig(savefilename,  bbox_inches="tight")
