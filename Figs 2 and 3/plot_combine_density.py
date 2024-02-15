import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm as gauss
from scipy.stats import bernoulli as bernoulli

from script.library.OT_utils import *

np.random.seed(seed= 2023)

n = 5000
alpha = 0.5

import scienceplots
plt.style.use(['science'])
import dill
with open('Script/Illustrative/mixtureG_result_4.pkl', 'rb') as f:
    running_time = dill.load(file=f)
    ot_dist = dill.load(file=f)
# bar plot
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.rcParams['text.usetex'] = True

fig = plt.figure(layout='constrained',figsize = (8,4))
subfig = fig.subfigures(nrows = 2, ncols = 1)

ax = {}
temp = subfig[0].subplots(1,4,sharey = False)
ax[0] = temp
temp = subfig[1].subplots(1,4,sharey = False)
ax[1] = temp
subfig[0].suptitle("Gamma",fontsize = 13)
subfig[1].suptitle("Gaussian mixture", fontsize = 13)


with plt.style.context(['science', 'ieee']):
    # Gamma:
    with open('Script/Illustrative/gamma_dist_result.pkl', 'rb') as f:
        a = dill.load(file=f)
        b = dill.load(file=f)
        c = dill.load(file=f)
        y_11_N_true = dill.load(file=f)
        nn_index_array = dill.load(file=f)
        CiC_counterfactual = dill.load(file=f)
        full_OT_counterfactual = dill.load(file=f)
        maxsw_sampling_counterfactual = dill.load(file=f)
        xmin = -8
        xmax = 13
        ymin = -8
        ymax = 13
        plot_density_ax(ax[0][0],y_11_N_true[0, :], y_11_N_true[1, :], title="Ground Truth",xtitle = 0.05,ytitle = 0.075, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                     x_size=13, y_size=13,xlabel = "")
        plot_density_ax(ax[0][1],full_OT_counterfactual[:, 0], full_OT_counterfactual[:, 1], title="OT", xtitle = 0.05,ytitle = 0.075,xmin=xmin, xmax=xmax,
                     ymin=ymin, ymax=ymax, x_size=13,xlabel = "",ylabel = "")
        plot_density_ax(ax[0][2],CiC_counterfactual[:, 0], CiC_counterfactual[:, 1], title="CiC",xtitle = 0.05,ytitle = 0.075, xmin=xmin, xmax=xmax, ymin=ymin,
                     ymax=ymax, x_size=13,xlabel = "",ylabel = "")
        # plot_density(sw_cf[:,0],sw_cf[:,1],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
        plot_density_ax(ax[0][3],maxsw_sampling_counterfactual[:, 0], maxsw_sampling_counterfactual[:, 1], title="Our method",xtitle = 0.05,ytitle = 0.075, xmin=xmin,
                     xmax=xmax, ymin=ymin, ymax=ymax, x_size=13,xlabel = "",ylabel = "")

    #Gaussian:
    # 2-D joint distribution
    with open('Script/Illustrative/mixtureG_dist_result_4.pkl', 'rb') as f:
        a = dill.load(file=f)
        b = dill.load(file=f)
        c = dill.load(file=f)
        y_11_N_true = dill.load(file=f)
        nn_index_array = dill.load(file=f)
        CiC_counterfactual = dill.load(file=f)
        full_OT_counterfactual = dill.load(file=f)
        maxsw_sampling_counterfactual = dill.load(file=f)
        xmin = -5
        xmax = 8
        ymin = -5
        ymax = 8
        plot_density_ax(ax[1][0],y_11_N_true[0,:],y_11_N_true[1,:],title = "Ground Truth",xtitle = 0.05,ytitle = 0.075,xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                 x_size=13,y_size=13)
        plot_density_ax(ax[1][1],full_OT_counterfactual[:,0],full_OT_counterfactual[:,1],title = "OT",xtitle = 0.1,ytitle = 0.075,xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
             x_size=13,ylabel = "")
        plot_density_ax(ax[1][2],CiC_counterfactual[:,0],CiC_counterfactual[:,1],title = "CiC",xtitle = 0.05,ytitle = 0.075,xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                 x_size=13,ylabel = "")
        plot_density_ax(ax[1][3],maxsw_sampling_counterfactual[:,0],maxsw_sampling_counterfactual[:,1],title = "Our method",xtitle = 0.05,ytitle = 0.075,xmin = xmin, xmax = xmax, ymin = ymin,
                  ymax = ymax,x_size=13,ylabel = "")


plt.subplots_adjust(wspace = 0.3, hspace=0)
fig.tight_layout()
fig.show()

fig.savefig("./images/manuscript/illustrative_density.png",dpi = 600)