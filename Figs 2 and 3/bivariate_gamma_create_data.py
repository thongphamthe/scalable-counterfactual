import numpy as np
import torch
from scipy.stats import gamma

from script.library.OT_utils import *

np.random.seed(seed= 2023)

n = 5000
alpha = 0.5

for jj in range(1):
# control group's hidden variable at time = 0
    v_1 = gamma.rvs(a = 2, size = n, scale = 3)
    v_2 = gamma.rvs(a = 3, size = n, scale = 2)
    v_time_0   = np.array([v_1, v_2]).transpose()
    # control group's hidden variable at time = 1
    v_1 = gamma.rvs(a = 2, size = n, scale = 3)
    v_2 = gamma.rvs(a = 3, size = n, scale = 2)
    v_time_1   = np.array([v_1, v_2]).transpose()
    # treatment group's hidden variable at time = 0
    v_star_1 = gamma.rvs(a = 3, size = n, scale = 2)
    v_star_2 = gamma.rvs(a = 2, size = n, scale = 3)
    v_star_time_0   = np.array([v_star_1, v_star_2]).transpose()
    # treatment group's hidden variable at time = 0
    v_star_1 = gamma.rvs(a = 3, size = n, scale = 2)
    v_star_2 = gamma.rvs(a = 2, size = n, scale = 3)
    v_star_time_1   = np.array([v_star_1, v_star_2]).transpose()
    # mapping function at time t = 0:
    h_0 = np.array([[1, alpha], [alpha, 1]])
    # a monotonic function in each argument
    #h_0 = np.array([[1, 0], [0, 1]])

    # mapping function at time t = 1:
    h_1 = np.array([[1, -alpha], [-alpha, 1]])
    #h_1 = np.array([[2, 0], [0, 2]])

    # time t = 0, control group:
    y_00_N = np.inner(h_0 ,v_time_0)

    # time t = 1, control group:
    y_01_N = np.inner(h_1 ,v_time_1)

    # time t = 0, treatment group:
    y_10_N = np.inner(h_0 ,v_star_time_0)

    # time t = 1, treatment group, not receive treatment
    # i.e., the counterfactual we need to estimate
    y_11_N_true = np.inner(h_1 ,v_star_time_1)


    a = torch.from_numpy(y_00_N.transpose())
    a = a.to(torch.float)
    b = torch.from_numpy(y_01_N.transpose())
    b = b.to(torch.float)
    c = torch.from_numpy(y_10_N.transpose())
    c = c.to(torch.float)

    nn_index_array = nearest_neighbor_index_finding(c, a)

    #_,sw_cf,time_3 = sliced_OT_causal_estimate(a,b,c,nn_index_array = nn_index_array,num_projs = 5)
    CiC_counterfactual,time_cic = marginal_OT_causal_estimate(a,b,c,nn_index_array = nn_index_array)
    maxsw_sampling_counterfactual,time_pp = MSW_by_sampling_causal_estimate(a,b,c,nn_index_array = nn_index_array,num_projs = 10)
    full_OT_counterfactual, time_ot = full_OT_causal_estimate(a,b,c,nn_index_array = nn_index_array, iter = 10000000, core = 1)

    emd_OT = emd_dist(full_OT_counterfactual,y_11_N_true.transpose())
    emd_cic = emd_dist(CiC_counterfactual,y_11_N_true.transpose())
    emd_proposed = emd_dist(maxsw_sampling_counterfactual,y_11_N_true.transpose())

import dill
with open('Script/Illustrative/gamma_dist_result.pkl', 'wb') as f:
    dill.dump(a,file = f)
    dill.dump(b,file = f)
    dill.dump(c, file=f)
    dill.dump(y_11_N_true, file = f)
    dill.dump(nn_index_array, file = f)
    dill.dump(CiC_counterfactual, file = f)
    dill.dump(full_OT_counterfactual, file = f)
    dill.dump(maxsw_sampling_counterfactual, file = f)




import dill
with open('Script/Illustrative/gamma_result.pkl', 'rb') as f:
    running_time = dill.load(file = f)
    ot_dist = dill.load(file = f)

# bar plot
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.rcParams['text.usetex'] = True
with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots()
    x_pos = ["OT","Proposed","CiC"]
    mean_value = [np.mean(running_time["OT"])]
    sd_value = [np.std(running_time["OT"])]
    mean_value.append(np.mean(running_time["proposed"]))
    mean_value.append(np.mean(running_time["CiC"]))
    sd_value.append(np.std(running_time["proposed"]))
    sd_value.append(np.std(running_time["CiC"]))

    ax.bar(x_pos, mean_value, yerr= sd_value, log=True,align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel("Running time (in milliseconds)")
    plt.tight_layout()
    plt.show()

    mean_value = [np.mean(ot_dist["OT"])]
    sd_value = [np.std(ot_dist["OT"])]
    mean_value.append(np.mean(ot_dist["proposed"]))
    mean_value.append(np.mean(ot_dist["CiC"]))
    sd_value.append(np.std(ot_dist["proposed"]))
    sd_value.append(np.std(ot_dist["CiC"]))
    fig, ax = plt.subplots()
    ax.bar(x_pos, mean_value, yerr=sd_value, log=True, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(r"OT distance to the empirical version of $\widetilde{\mu_{1}^{\texttt{T}}}$")
    plt.tight_layout()
    plt.show()



# 2-D joint distribution
xmin = -8
xmax = 13
ymin = -8
ymax = 13

plot_density(y_11_N_true[0,:],y_11_N_true[1,:],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,x_size=12,y_size=12,xticks = [-5,0,5,10],yticks = [-5,0,5,10])
plot_density(full_OT_counterfactual[:,0],full_OT_counterfactual[:,1],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,x_size=12,y_size=12)

plot_density(CiC_counterfactual[:,0],CiC_counterfactual[:,1],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,x_size=12,y_size=12)
#plot_density(sw_cf[:,0],sw_cf[:,1],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)
plot_density(maxsw_sampling_counterfactual[:,0],maxsw_sampling_counterfactual[:,1],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,x_size=12,y_size=12)

#plot_density(maxsw_optimization_counterfactual[:,0],maxsw_optimization_counterfactual[:,1],title = "",xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax)