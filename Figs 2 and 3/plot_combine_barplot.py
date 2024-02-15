import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm as gauss
from scipy.stats import bernoulli as bernoulli

from script.library.OT_utils import *

np.random.seed(seed= 2023)

n = 5000
alpha = 0.5

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# bar plot
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = True

font_size = 14
cap_size = 8

ax = {}
fig = plt.figure(layout='constrained',figsize = (9,2))
subfig = fig.subfigures(nrows = 1, ncols = 3,width_ratios=[3, 3,1], wspace = 0.2,hspace = 0.75)
temp = subfig[0].subplots(1,2)
ax[0] = temp
temp = subfig[1].subplots(1,2)
ax[1] = temp
ax[2] = subfig[2]
subfig[0].suptitle("Gamma",fontsize = font_size)
subfig[1].suptitle("Gaussian mixture", fontsize =font_size)


color = ["r","black","blue"]
alpha = 0.75
import dill
with open('Script/Illustrative/gamma_result.pkl', 'rb') as f:
    running_time = dill.load(file = f)
    ot_dist = dill.load(file = f)

with plt.style.context(['science', 'ieee']):

    x_pos = ["OT","CiC","Proposed"]
    mean_value = [np.mean(running_time["OT"])]
    mean_value.append(np.mean(running_time["CiC"]))
    mean_value.append(np.mean(running_time["proposed"]))

    sd_value = [np.std(running_time["OT"])]
    sd_value.append(np.std(running_time["CiC"]))
    sd_value.append(np.std(running_time["proposed"]))


    ax[0][0].bar(x_pos, mean_value,color = color, yerr= sd_value, log=True,align='center', alpha=alpha, ecolor='black', capsize=cap_size)
    ax[0][0].set_xlabel("",fontsize=font_size)
    ax[0][0].set_ylabel("Running time (ms)", fontsize=font_size)
    xticklabels = ax[0][0].get_xticklabels()
    yticklabels = ax[0][0].get_yticklabels()
    ax[0][0].set_xticklabels("", fontsize=font_size)
    ax[0][0].set_yticklabels(yticklabels, fontsize=font_size)


    #plt.show()

    mean_value = [np.mean(ot_dist["OT"])]
    mean_value.append(np.mean(ot_dist["CiC"]))
    mean_value.append(np.mean(ot_dist["proposed"]))

    sd_value = [np.std(ot_dist["OT"])]
    sd_value.append(np.std(ot_dist["CiC"]))
    sd_value.append(np.std(ot_dist["proposed"]))


    ax[0][1].bar(x_pos, mean_value,color = color, yerr=sd_value, log=True, align='center', alpha=alpha, ecolor= "black", capsize=cap_size)
    ax[0][1].set_xlabel("", fontsize=font_size)
    ax[0][1].set_ylabel("$OT(\cdot,\widetilde{\mu_{1}^{\\texttt{T}^*}})$", fontsize=font_size)
    xticklabels = ax[0][1].get_xticklabels()
    yticklabels = ax[0][1].get_yticklabels()
    ax[0][1].set_xticklabels("", fontsize=font_size)
    ax[0][1].set_yticklabels(yticklabels, fontsize=font_size)


with open('Script/Illustrative/mixtureG_result_4.pkl', 'rb') as f:
    running_time = dill.load(file=f)
    ot_dist = dill.load(file=f)

with plt.style.context(['science', 'ieee']):

    x_pos = ["OT","CiC","Proposed"]
    mean_value = [np.mean(running_time["OT"])]
    mean_value.append(np.mean(running_time["CiC"]))
    mean_value.append(np.mean(running_time["proposed"]))

    sd_value = [np.std(running_time["OT"])]
    sd_value.append(np.std(running_time["CiC"]))
    sd_value.append(np.std(running_time["proposed"]))


    ax[1][0].bar(x_pos, mean_value, color = color,yerr= sd_value, log=True,align='center', alpha=alpha, ecolor='black', capsize = cap_size)
    ax[1][0].set_xlabel("",fontsize=font_size)
    ax[1][0].set_ylabel("Running time (ms)", fontsize=font_size)
    xticklabels = ax[1][0].get_xticklabels()
    yticklabels = ax[1][0].get_yticklabels()
    ax[1][0].set_xticklabels("", fontsize=font_size)
    ax[1][0].set_yticklabels(yticklabels, fontsize=font_size)
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.9)

    #plt.show()

    mean_value = [np.mean(ot_dist["OT"])]
    mean_value.append(np.mean(ot_dist["CiC"]))
    mean_value.append(np.mean(ot_dist["proposed"]))

    sd_value = [np.std(ot_dist["OT"])]
    sd_value.append(np.std(ot_dist["CiC"]))
    sd_value.append(np.std(ot_dist["proposed"]))


    ax[1][1].bar(x_pos, mean_value,color = color, yerr=sd_value, log=True, align='center', alpha=alpha, ecolor='black', capsize = cap_size)
    ax[1][1].set_xlabel("", fontsize=font_size)
    ax[1][1].set_ylabel("$OT(\cdot,\widetilde{\mu_{1}^{\\texttt{T}^*}})$", fontsize=font_size)
    xticklabels = ax[1][1].get_xticklabels()
    yticklabels = ax[1][1].get_yticklabels()
    ax[1][1].set_xticklabels("", fontsize=font_size)
    ax[1][1].set_yticklabels(yticklabels, fontsize=font_size)

colors = {'OT':'red', 'CiC':'black',"Our method":"blue"}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),0.5,0.5, color=colors[label], alpha = alpha) for label in labels]
ax[2].legend(handles, labels,fontsize = 14,loc = "center",bbox_to_anchor = [0.25,0.5])
#plt.tight_layout()
#plt.subplots_adjust(hspace= 0,wspace=0.5)
plt.show()
fig.savefig("./images/manuscript/illustrative_barplot.png",dpi = 600)