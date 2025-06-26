import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_each as pe
from matplotlib.ticker import ScalarFormatter

colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
styles = ['solid', 'dotted', 'dashed', 'dashdot']
labelsize = 20
legendsize1 = 13
legendsize2 = 16
panellabelsize = 22

config = {
    'font.size': 14,
    'mathtext.fontset': 'cm',
}
plt.rcParams.update(config)

# read data file
filepass = './results/output_files_long_data'
param = pd.read_csv(filepass + "/true_parameters.csv")
df = pd.read_csv(filepass + "/data.csv")  

# parameters
K_true = param['K_true'][0]
gamma_true = param['gamma_true'][0]
time_cols = df.columns[2:]  # All columns except 'oscillator_number' and 'seed' contain time series data.
time_values = np.array([float(col) for col in df.columns[2:]])
nlags = 200
delta_t = 0.5 # sampling interval of order parameter dataset

# calculate autocorrelation function
def compute_acf(x, nlags):
    x_mean = np.mean(x)
    var = np.var(x)
    acf_vals = []
    for lag in range(nlags + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            cov = np.mean((x[:-lag] - x_mean) * (x[lag:] - x_mean))
            acf_vals.append(cov / var)
    return np.array(acf_vals)

# record data
acf_list = {}
ds = {}

for oscillator_number, group in df.groupby("oscillator_number"):
    row = group.iloc[0]
    series = row[time_cols].to_numpy().astype(float) # time series of the order parameter
    ds[oscillator_number] = series
    dif = series - pe.R_t(time_values, K_true, gamma_true) # deviation from the analytical solution
    dif = dif[10000:] # use only after t = 5000
    acf = compute_acf(dif, nlags)
    acf_list[oscillator_number] = acf

# plot Figure 5
fig = plt.figure(figsize=(6, 9))
ax1 = fig.add_subplot(2,1,1)
for i, oscillator_number in enumerate(sorted(ds)):
    series = ds[oscillator_number]
    labelstr = pe.create_label(oscillator_number)
    ax1.plot(time_values, series, label=labelstr,
             color=colors[i], linestyle=styles[i], lw=2)
ax1.plot(time_values, pe.R_t(time_values, K_true, gamma_true),
         label=r"$R_{\mathrm{sol}}(t;\theta)$",
         color=colors[-1], linestyle=styles[-1], lw=2)
ax1.set_xlabel(r"$t$", fontsize=labelsize)
ax1.set_xlim([9800, 10000])
ax1.set_ylim([-0.01, 0.21])
ax1.set_ylabel(r"$R(t)$", fontsize=labelsize)
ax1.set_yticks([0.0, 0.1, 0.2])
ax1.set_xticks([9800, 9900, 10000])
ax1.text(-0.02, 1.12, '(a)', horizontalalignment='right', verticalalignment='top', transform=ax1.transAxes, fontsize=panellabelsize)
ax1.legend(ncol=2, fontsize=legendsize1)

ax2 = fig.add_subplot(2,1,2)
pos = ax2.get_position()
inset_ax2 = fig.add_axes([pos.x0 + 0.16, pos.y0 + 0.15, 0.22, 0.15])

for i, oscillator_number in enumerate(sorted(acf_list)):
    acf = acf_list[oscillator_number]
    lags = np.arange(len(acf))*delta_t
    labelstr = pe.create_label(oscillator_number)
    ax2.plot(lags, acf, label=labelstr,
             color=colors[i], linestyle=styles[i], lw=4)
    inset_ax2.plot(lags, acf, 
                   color=colors[i], linestyle=styles[i], lw=3)

ax2.set_xlabel(r"$\tau$", fontsize=labelsize)
ax2.set_ylabel(r"$C(\tau$)", fontsize=labelsize)
ax2.set_xlim([0,100])
ax2.legend(fontsize=legendsize2)
ax2.text(-0.02, 1.12, '(b)', horizontalalignment='right', verticalalignment='top', transform=ax2.transAxes, fontsize=panellabelsize)
inset_ax2.set_xlim([0,20])
inset_ax2.set_ylim([0.04,1])
inset_ax2.set_yscale('log')
inset_ax2.set_yticks([0.1, 1.0])
inset_ax2.set_yticklabels(["0.1", "1.0"])

plt.subplots_adjust(hspace=0.4)
plt.savefig("long_data.pdf", bbox_inches='tight')