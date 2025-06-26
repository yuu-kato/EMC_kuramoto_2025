'''
Functions to plot figures from single EMC simulation data. 
'''

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle
from sklearn.linear_model import LinearRegression

### General setting for figures
mygray = [0.35, 0.35, 0.35]
myblue = '#005AFF'
myorange = '#F6AA00'
myred = '#FF4B00'

panellabelsize = 20
paneltitlesize = 22
panellabelsize2 = 30
labelsize = 20
legendsize = 18
config = {
    'font.size': 13,
    # 'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'cm',
}
plt.rcParams.update(config)

# settings for histgrams
wid_ratio = 3.0
bin_num = 201

### Functions
# Extract data
def read_data(filepass):
    # true parameters
    param = pd.read_csv(filepass + "/true_parameters.csv")

    # data points
    t, R= np.loadtxt(filepass + "/data.txt", usecols = [0, 1], unpack=True) 

    # free energy
    b, f= np.loadtxt(filepass + "/free_energy.txt", usecols = [0, 1], unpack=True)
    index = np.argmin(f)

    # sampling
    sample = pd.read_csv(filepass + "/parameter.csv")
    sample_burn = sample[sample['mc_step'] > param['burn_in'][0]]

    # MAP
    MAP = pd.read_csv(filepass + "/MAP.csv")
    
    # estimation error
    est_error = pd.read_csv(filepass + "/error.csv")

    return param, t, R, b, f, index, sample_burn, MAP, est_error

# create label for large oscillator numbers
def create_label(oscillator_number):
    if oscillator_number >= 1e5:
        sci_str = "{:.0e}".format(oscillator_number)
        base, exp = sci_str.split('e')
        exp = int(exp)
        labelstr = rf'$N=10^{{{exp}}}$'
    else:
        labelstr = rf'$N={oscillator_number}$'
    return labelstr

# Fitting
def R_t(t, K, gamma):
    R_0 = 1.0
    R = np.exp(-gamma * t + 0.5 * K * t) / np.sqrt(1.0/R_0/R_0 + (np.exp(-2.0 * gamma * t + K * t) - 1.0) * K / (- 2 * gamma + K))
    return R

def plot_data(t, R, MAP, param, index, fig, gs, label, ylabel, label_sum, mode):
    gamma_estimated = MAP['%s' % index][0]
    K_estimated = 2 * (gamma_estimated - MAP['%s' % index][1])
    
    if mode=='OA':
        labelstr = label_sum + rf' $\sigma={param['sigma'][0]}$'
        ylim = [-0.12, 1.2]
    elif mode=='kuramoto':
        osci_num = param['M'][0]
        ylim = [-0.02, 1.05]
        labelstr = create_label(osci_num)

    # plot
    ax = fig.add_subplot(gs[0,0])
    ax.plot(t, R_t(t, K_estimated, gamma_estimated), color = 'black', lw = 3, label = 'fitting')
    ax.scatter(t, R, color = mygray, label = 'data', s=25)
    ax.set_xlabel(r'$t$', fontsize = labelsize)
    ax.set_ylim(ylim)
    ax.text(-0.1, 1.14, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.text(0.5, 1.28, labelstr, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize2)
    if ylabel==True:
        ax.set_ylabel(r'$R(t)$', fontsize = labelsize)
        ax.legend(fontsize = legendsize)

# Free energy
def plot_free(b, f, index, param, fig, gs, label, ylabel, mode):
    xlim = [b[index]*0.1, min(b[index]*2.4, param['b_max'][0])]
    ylim = [f[index]-5, f[index]+60]

    if mode=='OA':
        sigma_infer = (b[index])**(-0.5)
        titlestr = rf'$\hat{{\sigma}}={round(sigma_infer,4)},\: \hat{{l}}={index+1}$'
    elif mode=='kuramoto':
        titlestr = rf'$\hat{{l}}={index+1}$'
                                                        
    ax = fig.add_subplot(gs[0,0])
    ax.plot(b, f, color = 'black', lw=3, marker=MarkerStyle("s"), markersize=12, markeredgecolor='black', markeredgewidth = 3, markerfacecolor = 'white', label = r'$F(b_l)$')
    if mode=='OA':
        sigma_true = param['sigma'][0]
        ax.vlines(sigma_true**(-2), -2000, 2000, color = myblue, linestyle = 'solid', lw=3, label= 'true')
    ax.vlines(b[index], -2000, 2000, color = myorange, linestyle = 'dotted', lw=6, label=r'infer $(b_\hat{l})$')
    ax.text(-0.1, 1.14, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Inverse temperature', fontsize = labelsize)
    ax.ticklabel_format(axis="x", scilimits=(-2, 3), useMathText=True)
    ax.set_title(titlestr, horizontalalignment='right', verticalalignment='bottom', fontsize=paneltitlesize, x=1.0, y=1.0)
    if ylabel==True:
        ax.set_ylabel('Free energy', fontsize = labelsize)
        ax.legend(fontsize = legendsize)

# Determine ylim for 1d histgram
def ylim_1d(mode):
    if mode=='OA':
        ylim = [0, 210]
    elif mode=='kuramoto':
        ylim = [0, 410]
    return ylim

# Histgrams
def plot_K(sample_burn, index, param, MAP, fig, gs, label, ylabel, mode):
    # preparation 
    sample_K = 2 * (sample_burn['l = %s_gamma' % index] - sample_burn['l = %s_delta_gamma' % index])
    K_true = param['K_true'][0]
    K_estimated = 2 * (MAP['%s' % index][0] - MAP['%s' % index][1])
    K_std = np.std(sample_K)

    xlim_min = K_estimated - wid_ratio * K_std
    xlim_max = K_estimated + wid_ratio * K_std
    xlim = [xlim_min, xlim_max]
    bin_list = np.linspace(xlim_min, xlim_max, bin_num)

    # plot
    ax = fig.add_subplot(gs[0,0])
    ax.hist(sample_K, bins = bin_list, histtype = 'step', color = 'black', lw = 1.3)
    ax.set_xlabel(r'$K$', fontsize = labelsize)
    ax.axvline(K_true, color = myblue, lw = 3, linestyle = 'solid', label = 'true')
    ax.axvline(K_estimated, color = myorange, lw = 6, linestyle = 'dotted', label = r'$\hat{K}$')
    ax.axvspan(K_estimated - K_std, K_estimated + K_std, color=myred, alpha = 0.1, label=rf'$\pm 1 \ \mathrm{{S.D.}}$')
    ax.set_xlim(xlim)
    # ax.set_ylim(ylim_1d(mode))
    ax.ticklabel_format(axis="y", scilimits=(-1, 3), useMathText=True)
    # ax.ticklabel_format(axis="x", scilimits=(-2, 3), useMathText=True)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.text(-0.1, 1.18, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.set_title(rf'$\hat{{K}}= {round(K_estimated, 4)}$', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=paneltitlesize, x=1.0, y=1.0)
    if ylabel==True:
        ax.set_ylabel('Number of samples', fontsize = labelsize)
        ax.legend(fontsize = legendsize)

def plot_gamma(sample_burn, index, param, MAP, fig, gs, label, ylabel, mode):
    # preparation 
    sample_gamma = sample_burn['l = %s_gamma' % index] 
    gamma_true = param['gamma_true'][0]
    gamma_estimated = MAP['%s' % index][0]
    gamma_std = np.std(sample_gamma)

    xlim_min = gamma_estimated - wid_ratio * gamma_std
    xlim_max = gamma_estimated + wid_ratio * gamma_std
    xlim = [xlim_min, xlim_max]
    bin_list = np.linspace(xlim_min, xlim_max, bin_num)

    # plot
    ax = fig.add_subplot(gs[0,0])
    ax.hist(sample_gamma, bins = bin_list, histtype = 'step', color = 'black', lw = 1.3)
    ax.set_xlabel(r'$\gamma$', fontsize = labelsize)
    ax.axvline(gamma_true, color = myblue, lw = 3, linestyle = 'solid', label = 'true')
    ax.axvline(gamma_estimated, color = myorange, lw = 6, linestyle = 'dotted', label = r'$\hat{\gamma}$')
    ax.axvspan(gamma_estimated - gamma_std, gamma_estimated + gamma_std, color=myred, alpha = 0.1, label = rf'$\pm 1\ \mathrm{{S.D.}}$')
    ax.set_xlim(xlim)
    ax.ticklabel_format(axis="y", scilimits=(-1, 3), useMathText=True)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.text(-0.1, 1.18, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.set_title(rf'$\hat{{\gamma}}={round(gamma_estimated, 4)}$', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=paneltitlesize, x=1.0, y=1.0)
    if ylabel==True:
        ax.set_ylabel('Number of samples', fontsize = labelsize)
        ax.legend(fontsize = legendsize)

def plot_2d(sample_burn, index, param, MAP, fig, gs, label, ylabel, cbar):
    # preparation 
    sample_K = 2 * (sample_burn['l = %s_gamma' % index] - sample_burn['l = %s_delta_gamma' % index])
    sample_gamma = sample_burn['l = %s_gamma' % index] 
    K_true = param['K_true'][0]
    gamma_true = param['gamma_true'][0]
    x=sample_K.to_list()
    y=sample_gamma.to_list()
    K_estimated = 2 * (MAP['%s' % index][0] - MAP['%s' % index][1])
    K_std = np.std(sample_K)
    gamma_estimated = MAP['%s' % index][0]
    gamma_std = np.std(sample_gamma)

    # settings for figure
    xlim_min = K_estimated - wid_ratio * K_std
    xlim_max = K_estimated + wid_ratio * K_std
    xlim = [xlim_min, xlim_max]
    ylim_min = gamma_estimated - wid_ratio * gamma_std
    ylim_max = gamma_estimated + wid_ratio * gamma_std
    ylim = [ylim_min, ylim_max]

    bins_x = np.linspace(xlim_min, xlim_max, bin_num)
    bins_y = np.linspace(ylim_min, ylim_max, bin_num)
    
    # linear regression
    x_reg = np.array(x).reshape(-1, 1)
    y_reg = np.array(y).reshape(-1, 1)
    mask = (xlim_min < x_reg) & (x_reg < xlim_max) & (ylim_min < y_reg) & (y_reg < ylim_max) 
    x_reg = x_reg[mask].reshape(-1, 1) # remove the outliers
    y_reg = y_reg[mask].reshape(-1, 1) # remove the outliers
    model = LinearRegression()
    model.fit(x_reg,y_reg)
    a = model.coef_.item() # slope
    b = model.intercept_.item() # y-intercept

    # main heatmap
    ax_main = fig.add_subplot(gs[:, :])
    h = ax_main.hist2d(x, y, bins=(bins_x, bins_y), cmap='viridis', norm=LogNorm(vmin=1e0, vmax=5e2))
    ax_main.axhline(y=gamma_true, color='gray', linestyle='--', linewidth=1)
    ax_main.axvline(x=K_true, color='gray', linestyle='--', linewidth=1)
    ax_main.plot(bins_x, a*bins_x + b, color='gray')
    ax_main.set_xlabel(r'$K$', fontsize = labelsize)
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.get_xaxis().set_tick_params(pad=10)
    ax_main.ticklabel_format(axis="y", scilimits=(-2, 3), useMathText=True)
    ax_main.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax_main.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax_main.text(-0.01, 1.12, label, horizontalalignment='center', verticalalignment='top', transform=ax_main.transAxes, fontsize=panellabelsize)
    ax_main.text(0.65, 0.15, rf'$\gamma={round(a,3)}K+{round(b,4)}$', horizontalalignment='center', verticalalignment='top', transform=ax_main.transAxes, fontsize=panellabelsize)

    if ylabel==True:
        ax_main.set_ylabel(r'$\gamma$', fontsize = labelsize)

    if cbar==True:
        cbar = fig.colorbar(h[3], pad=0.03)
        cbar.ax.tick_params(labelsize=13) 
    
# plot esimation error
def plot_error(param, est_error, label_sum, fig, i, mode):    
    # settings
    if mode=='OA':
        labelstr = label_sum + rf' $\sigma={param['sigma'][0]}$'
        ylim = [param['true_error'][0]*0.7, 1e-1]
    elif mode=='kuramoto':
        osci_num = param['M'][0]
        ylim = [param['true_error'][0]*0.5, 1e-1]
        labelstr = create_label(osci_num)
    
    # mean and SE of independent simulations
    mean_err = est_error.mean(axis=0)[1:]
    mean_err = np.array(mean_err.tolist())
    std_err = est_error.std(axis=0, ddof=0)[1:]
    std_err = np.array(std_err.tolist())
    ste_err = std_err/np.sqrt(len(est_error)) # standard error
    ste_err = np.array(ste_err.tolist())
    mc_step = [float(s) for s in est_error.columns.tolist()[1:]]

    # plot
    ax = fig.add_subplot(3,1,(i+1))
    ax.scatter(mc_step[0], mean_err[0], marker='x', s=50, linewidths=2, c=myblue, label='initial mean error')
    ax.plot(mc_step, mean_err, lw=1.3, color=myblue, label='mean error')
    ax.axhline(y=param['true_error'][0], color=myred, linestyle=(0, (1, 1)), linewidth=3, label='true error') # densely dotted
    ax.fill_between(
        mc_step,
        mean_err - ste_err,
        mean_err + ste_err,
        color=myblue, 
        alpha=0.3,
        label='SE of error'
    )
    ax.set_yscale('log')
    ax.text(0.65, 0.95, labelstr, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=20)
    ax.set_xlim([-300, 20000])
    ax.set_ylim(ylim)
    
    if i == 0:
        ax.legend(loc='upper left', bbox_to_anchor = (0.05, 1.03), fontsize = 13)
