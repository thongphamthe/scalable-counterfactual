import numpy.random
import requests
import zipfile
import pandas as pd
from matplotlib import pyplot as plt
from io import BytesIO, TextIOWrapper
import numpy as np
import torch
from script.library.OT_utils import *
from sklearn.utils import Bunch
from mlxtend.preprocessing import standardize

import matplotlib.pyplot as plt
import scienceplots

numpy.random.seed(1)

r = requests.get("https://davidcard.berkeley.edu/data_sets/njmin.zip")
z = zipfile.ZipFile(BytesIO(r.content))
df = pd.read_csv(z.open("public.dat"), engine="python", sep="\s+",
                  header=None).applymap(lambda x: pd.to_numeric(x, errors="coerce"))
# Load column names and descriptions from `codebook`
codebook = [repr(line)
             for line in TextIOWrapper(z.open("codebook"), "cp437")]
# part of the codebook is not relevant
codes = codebook[7:11] + codebook[13:19] + \
         codebook[21:38] + codebook[40:59]
cols = [i.strip("'\" ").split()[0] for i in codes]
descriptions = [" ".join(i.strip("'\" ").split()[
                          4:]).rstrip('\\n') for i in codes]
column_descriptions = dict(zip(cols, descriptions))
df.columns = cols



# full time employment: df.EMPFT (before) and df.EMPFT2 (after)
# part time employment df.EMPPT (before) and df.EMPPT2 (after)
# control group: df.STATE = 0 (PA), treatment group: df.STATE = 1 (NJ)
y_00_N_ori = np.array([df.EMPFT[df.STATE == 0].to_numpy(),
                   df.EMPPT[df.STATE == 0].to_numpy(),
                  df.HRSOPEN[df.STATE == 0].to_numpy(),
                   df.OPEN[df.STATE == 0].to_numpy(),
                   df.NMGRS[df.STATE == 0].to_numpy(),
                   df.NREGS[df.STATE == 0].to_numpy(),
                   df.INCTIME[df.STATE == 0].to_numpy(),
                   df.PSODA[df.STATE == 0].to_numpy(),
                   df.PENTREE[df.STATE == 0].to_numpy()
                   ]).transpose()

y_01_N_ori = np.array([df.EMPFT2[df.STATE == 0].to_numpy(),
                   df.EMPPT2[df.STATE == 0].to_numpy(),
                   df.HRSOPEN2[df.STATE == 0].to_numpy(),
                   df.OPEN2R[df.STATE == 0].to_numpy(),
                   df.NMGRS2[df.STATE == 0].to_numpy(),
                   df.NREGS2[df.STATE == 0].to_numpy(),
                   df.INCTIME2[df.STATE == 0].to_numpy(),
                   df.PSODA2[df.STATE == 0].to_numpy(),
                   df.PENTREE2[df.STATE == 0].to_numpy()
                   ]).transpose()




y_10_N_ori = np.array([df.EMPFT[df.STATE == 1].to_numpy(),
                   df.EMPPT[df.STATE == 1].to_numpy(),
                   df.HRSOPEN[df.STATE == 1].to_numpy(),
                   df.OPEN[df.STATE == 1].to_numpy(),
                   df.NMGRS[df.STATE == 1].to_numpy(),
                   df.NREGS[df.STATE == 1].to_numpy(),
                   df.INCTIME[df.STATE == 1].to_numpy(),
                   df.PSODA[df.STATE == 1].to_numpy(),
                   df.PENTREE[df.STATE == 1].to_numpy()
                   ],
                  ).transpose()

y_11_N_ori = np.array([df.EMPFT2[df.STATE == 1].to_numpy(),
                   df.EMPPT2[df.STATE == 1].to_numpy(),
                   df.HRSOPEN2[df.STATE == 1].to_numpy(),
                   df.OPEN2R[df.STATE == 1].to_numpy(),
                   df.NMGRS2[df.STATE == 1].to_numpy(),
                   df.NREGS2[df.STATE == 1].to_numpy(),
                   df.INCTIME2[df.STATE == 1].to_numpy(),
                   df.PSODA2[df.STATE == 1].to_numpy(),
                   df.PENTREE2[df.STATE == 1].to_numpy()
                   ]).transpose()

choose_variable = [0,1,2,3,4,5,6,7,8]
y_00_N = y_00_N_ori[:,choose_variable]
y_01_N = y_01_N_ori[:,choose_variable]
y_10_N = y_10_N_ori[:,choose_variable]
y_11_N = y_11_N_ori[:,choose_variable]



nan_control_0 = np.where(np.isnan(y_00_N))[0]
nan_control_1 = np.where(np.isnan(y_01_N))[0]
y_00_N = np.delete(y_00_N,np.concatenate([nan_control_0,nan_control_1]),0)
y_01_N = np.delete(y_01_N,np.concatenate([nan_control_0,nan_control_1]),0)

nan_0 = np.where(np.isnan(y_10_N))[0]
nan_1 = np.where(np.isnan(y_11_N))[0]
y_10_N = np.delete(y_10_N,np.concatenate([nan_0,nan_1]),0)
y_11_N = np.delete(y_11_N,np.concatenate([nan_0,nan_1]),0)

standardized = False
if standardized:
    y_00_N = standardize(y_00_N)
    y_01_N = standardize(y_01_N)
    y_10_N = standardize(y_10_N)
    y_11_N = standardize(y_11_N)

a = torch.from_numpy(y_00_N)
a = a.to(torch.float)
b = torch.from_numpy(y_01_N)
b = b.to(torch.float)
c = torch.from_numpy(y_10_N)
c = c.to(torch.float)
d = torch.from_numpy(y_11_N)
d = d.to(torch.float)

nn_index_array = nearest_neighbor_index_finding(c, a)


full_OT_counterfactual, time_2 = full_OT_causal_estimate(a,b,c,nn_index_array = nn_index_array, iter = 10000000, core = 1)
CiC_counterfactual,time_1 = marginal_OT_causal_estimate(a,b,c,nn_index_array = nn_index_array)

print("CiC emd: ",emd_dist(CiC_counterfactual,full_OT_counterfactual))

emd_array_samp  = []
emd_array_time = []

for jj in range(1000):
    maxsw_sampling_counterfactual, time_5 = MSW_by_sampling_causal_estimate(a, b, c, nn_index_array=nn_index_array,num_projs=10)
    emd_array_samp.append(emd_dist(full_OT_counterfactual,maxsw_sampling_counterfactual))
    emd_array_time.append(time_5)


print("EMD mean and std of the mean:")
print(np.mean(emd_array_samp))
print(np.std(emd_array_samp)/np.sqrt(1000))


if not standardized:
    max_x = 35
    min_x = -1
    max_y = 50
    min_y = -1
else:
    max_x = 6
    min_x = -3
    min_y = -3
    max_y = 4



size = 11

##### plotting counterfactual

import scienceplots
import numpy as np

plt.style.use(['science','no-latex'])
plt.rcParams['text.usetex'] = True


font_size = 10
fig, ax = plt.subplots(1,3,
                       subplot_kw=dict(aspect = "equal"),
                       figsize = (6.6,2.20),
                       sharey=False
                       )

#ax[0].scatter(diff_full_OT[:,0],diff_full_OT[:,1])
ax[0].scatter(full_OT_counterfactual[:,0],full_OT_counterfactual[:,1],size)
#ax[0].set_aspect('equal')
ax[0].set_xlim([min_x,max_x])
ax[0].set_ylim([min_y,max_y])
ax[0].set_xlabel("Full-time",fontsize = font_size)
ax[0].set_ylabel("Part-time",fontsize = font_size)
xticklabels = ax[0].get_xticklabels()
yticklabels = ax[0].get_yticklabels()
ax[0].set_xticklabels(xticklabels,fontsize = font_size) #fontsize = 14
ax[0].set_yticklabels(yticklabels,fontsize = font_size)


ax[0].set_title('OT',fontsize = font_size)

# CiC
ax[1].scatter(CiC_counterfactual[:,0],CiC_counterfactual[:,1],size)

ax[1].set_xlim([min_x,max_x])
ax[1].set_ylim([min_y,max_y])
ax[1].set_xlabel("Full-time",fontsize = font_size)

xticklabels = ax[1].get_xticklabels()
yticklabels = ax[1].get_yticklabels()
ax[1].set_xticklabels(xticklabels,fontsize = font_size)
ax[1].set_yticklabels(yticklabels,fontsize = font_size)

ax[1].set_title('CiC',fontsize = font_size)

ax[2].set_xlim([min_x,max_x])
ax[2].set_ylim([min_y,max_y])
ax[2].scatter(maxsw_sampling_counterfactual[:,0],maxsw_sampling_counterfactual[:,1],size)

ax[2].set_xlabel("Full-time",fontsize = font_size)

xticklabels = ax[2].get_xticklabels()
yticklabels = ax[2].get_yticklabels()
ax[2].set_xticklabels(xticklabels,fontsize = font_size)
ax[2].set_yticklabels(yticklabels,fontsize = font_size)

ax[2].set_title("Our method",fontsize = font_size)

plt.subplots_adjust(left=0,
                    bottom=0.2,
                    right=1,
                    top=0.9,hspace = 0,wspace = 0)
#fig.tight_layout()
fig.show()
fig.savefig("./images/manuscript/CK_result.png",dpi = 600)
