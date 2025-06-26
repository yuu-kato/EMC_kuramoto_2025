import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import plot_each as pe

from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle

### Common variables
# Names of panels
label_sum_list = ["(A)", "(B)", "(C)"]
label_list = ["(A-1)", "(A-2)", "(A-3)", "(A-4)", "(A-5)", "(B-1)", "(B-2)", "(B-3)", "(B-4)", "(B-5)", "(C-1)", "(C-2)", "(C-3)", "(C-4)", "(C-5)"]

### Functions to draw whole figure
# create grid specs
def create_gs():
    gs_list = []
    x_begin = 0.03 # left margin
    y_begin = 1 - 0.05 # top margin
    x_width = 0.26 # width of each panel
    y_width = 0.15 # height of each panel
    y_width1 = 0.12 # height of 1d hist
    y_width2 = 0.21 # height of 2d hist.
    x_inter = 0.05
    y_inter = 0.06
    y_inter1 = 0.06
    y_inter2 = 0.05

    x = x_begin
    y = y_begin
    for i in range(3):
        for j in range(5):
            if j==4:
                gs = gridspec.GridSpec(5, 5, left=x, right= x + x_width, bottom = y-y_width2, top = y, wspace=0.05, hspace=0.06)
            elif j==2 or j==3:
                gs = gridspec.GridSpec(1, 1, left=x, right= x + x_width, bottom = y-y_width1, top = y)
                if j==3:
                    y -= y_width1 + y_inter1
                else:
                    y -= y_width1 + y_inter2
            else:
                gs = gridspec.GridSpec(1, 1, left=x, right= x + x_width, bottom = y-y_width, top = y)
                y -= y_width + y_inter
            gs_list.append(gs)
        x += x_width + x_inter
        y = y_begin
    return gs_list

# draw figure
def draw_all(mode, file_list, savefilename):
    fig = plt.figure(figsize=(18, 18))

    # prepare grid specs
    gs_list = create_gs()

    # draw each panel
    fig_index = 0
    ylabel_list = [True, False, False]
    cbar_list = [False, False, True]
    for i in range(3):
        param, t, R, b, f, index, sample_burn, MAP, est_error = pe.read_data(file_list[i])
        ylabel = ylabel_list[i]
        cbar = cbar_list[i]
        for j in range(5):
            gs = gs_list[fig_index]
            label = label_list[fig_index]
            label_sum = label_sum_list[i]
            if j==0:
                pe.plot_data(t, R, MAP, param, index, fig, gs, label, ylabel, label_sum, mode)
            elif j==1:
                pe.plot_free(b, f, index, param, fig, gs, label, ylabel, mode)
            elif j==2:
                pe.plot_K(sample_burn, index, param, MAP, fig, gs, label, ylabel, mode)
            elif j==3:
                pe.plot_gamma(sample_burn, index, param, MAP, fig, gs, label, ylabel, mode)
            elif j==4:
                pe.plot_2d(sample_burn, index, param, MAP, fig, gs, label, ylabel, cbar)
            fig_index += 1
    plt.savefig(savefilename, bbox_inches='tight')
    
# plot estimation error
def draw_error(mode, file_list, savefilename):
    fig = plt.figure(figsize=(6, 12))
    for i in range(3):
        param, t, R, b, f, index, sample_burn, MAP, est_error = pe.read_data(file_list[i])
        label_sum = label_sum_list[i]
        pe.plot_error(param, est_error, label_sum, fig, i, mode)
    fig.supxlabel('MC steps', y = 0.05, fontsize = 20)
    fig.supylabel('Estimation error ' + r'$E(\theta)$', x = -0.03, fontsize = 25)
    plt.savefig(savefilename, bbox_inches='tight')
    
# plot analytical curves
def draw_analy_sol(savefilename):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1,1,1)
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
    
    t = np.linspace(0, 50, 1001)
    K = np.arange(0.03, 0.07, 0.01)
    gamma = 0.33*K + 0.0635
    
    for i in range(len(K)):
        R_series = pe.R_t(t, K[i], gamma[i])
        if i==2:
            ax.plot(t, R_series, 
                label=rf'$K_{{\mathrm{{true}}}}={round(K[i],2)},\ \gamma_{{\mathrm{{true}}}}={round(gamma[i],4)}$',
                color=colors[i], lw = 1)
        else:
            ax.plot(t, R_series, 
                label=rf'$K={round(K[i],2)}, \gamma={round(gamma[i],4)}$',
                color=colors[i], lw = 1)
    
    ax.set_xlabel(r'$t$', fontsize = 20)
    ax.set_ylabel(r'$R_{\mathrm{sol}}(t;\theta)$', fontsize = 20)
    ax.legend()
    plt.savefig(savefilename, bbox_inches='tight')

### Figure 1, 6, 8: OA
mode = 'OA'

# file settings
file_list = ["./results/output_files_OA_sigma01", "./results/output_files_OA_sigma001", "./results/output_files_OA_sigma0001"]

# draw results (Fig. 1)
savefilename = "OA_single.pdf"
draw_all(mode, file_list, savefilename)

# draw estimation error (Fig. 6)
file_list = ["./results/output_files_OA_sigma01_error", "./results/output_files_OA_sigma001_error", "./results/output_files_OA_sigma0001_error"]
savefilename = "OA_error.pdf"
draw_error(mode, file_list, savefilename)

# draw analytical curves for different K, gamma (Fig. 8)
savefilename = "correlations.pdf"
draw_analy_sol(savefilename)

### Figure 3, 7: kuramoto
mode = 'kuramoto'

# file settings
file_list = ["./results/output_files_kuramoto_n100", "./results/output_files_kuramoto_n1000", "./results/output_files_kuramoto_n100000"]

# draw results (Fig. 3)
savefilename = "kuramoto_single.pdf"
draw_all(mode, file_list, savefilename)

# draw estimation error (Fig. 7)
file_list = ["./results/output_files_kuramoto_n100_error", "./results/output_files_kuramoto_n1000_error", "./results/output_files_kuramoto_n100000_error"]
savefilename = "kuramoto_error.pdf"
draw_error(mode, file_list, savefilename)