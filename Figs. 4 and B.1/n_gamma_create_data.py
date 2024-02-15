import numpy as np
import torch
from scipy.stats import gamma

from script.library.OT_utils import *

np.random.seed(seed= 2023)

n_array        = [5000,10000,15000,20000]
num_iter_array = [1,2,10,50,100,200,500]
m       = 10
alpha   = 0.5



run_vector = ["full","CiC","maxSW_sampling","maxSW_adam","SW"]

#

def create_data(num_sample):
    import dill
    a1 = 2;
    scale1 = 3;
    a2 = 3;
    scale2 = 2;
    a3 = 3;
    scale3 = 2;
    a4 = 2;
    scale4 = 3;
    alpha = 0.5;
    m = 10
    h_0 = np.array([[1, alpha], [alpha, 1]])
    h_1 = np.array([[1, -alpha], [-alpha, 1]])
    for run_number in range(m):
        v_time_0,v_time_1, v_star_time_0,v_star_time_1 = create_data_gamma(num_sample, a1, scale1, a2, scale2, a3, scale3, a4, scale4)
        y_00_N = np.inner(h_0, v_time_0)

        # time t = 1, control group:
        y_01_N = np.inner(h_1, v_time_1)

        # time t = 0, treatment group:
        y_10_N = np.inner(h_0, v_star_time_0)

        # time t = 1, treatment group, not receive treatment
        # i.e., the counterfactual we need to estimate
        y_11_N_true = np.inner(h_1, v_star_time_1)
        a = torch.from_numpy(y_00_N.transpose())
        a = a.to(torch.float)
        b = torch.from_numpy(y_01_N.transpose())
        b = b.to(torch.float)
        c = torch.from_numpy(y_10_N.transpose())
        c = c.to(torch.float)

        nn_index_array = nearest_neighbor_index_finding(c, a)


        with open('data_file/n_gamma/n_gamma_n_' + str(num_sample) + "_num_" + str(run_number) + '.pkl', 'wb') as f:
            dill.dump(h_0, file=f)
            dill.dump(h_1, file=f)
            dill.dump(v_time_0, file=f)
            dill.dump(v_time_1, file=f)
            dill.dump(v_star_time_0, file=f)
            dill.dump(v_star_time_1, file=f)
            dill.dump(m,file = f)
            dill.dump(alpha, file=f)
            dill.dump(a1, file=f)
            dill.dump(scale1, file=f)
            dill.dump(a2, file=f)
            dill.dump(scale2, file=f)
            dill.dump(a3, file=f)
            dill.dump(scale3, file=f)
            dill.dump(a4, file=f)
            dill.dump(scale4, file=f)
            dill.dump(y_00_N, file=f)
            dill.dump(y_01_N, file=f)
            dill.dump(y_10_N, file=f)
            dill.dump(y_11_N_true, file=f)
            dill.dump(a,file = f)
            dill.dump(b, file=f)
            dill.dump(c, file=f)
            dill.dump(nn_index_array, file=f)
            f.close()


import joblib
result = joblib.Parallel(n_jobs=len(n_array))(joblib.delayed(create_data)(n_sample) for n_sample in n_array)

