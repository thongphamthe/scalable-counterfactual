import numpy as np
import torch
from scipy.stats import gamma
import dill
from script.library.OT_utils import *
from numpy.linalg import solve

np.random.seed(seed= 2023)


n = 5000

m = 10
num_of_repeat = 20
dim_array = [2,10,50,100]
num_iter_array = [1,2,4,8,16,32,64,128,256]


def create_data(dim):
    h_0 = np.random.rand(dim, dim)
    np.fill_diagonal(h_0, 1)
    B = np.diag(np.random.rand(dim))
    h_1 = solve(h_0.transpose(), B)

    #print(h_0)
    #print(h_1)


    #np.all(np.linalg.eigvals(np.inner(h_0.transpose(),h_1)) > 0)

    time_dict = {}
    emd_dict = {}
    for run_number in range(m):
        v_time_0       = np.zeros((n,dim))
        v_star_time_0  = np.zeros((n,dim))
        v_time_1 = np.zeros((n, dim))
        v_star_time_1 = np.zeros((n, dim))
        for d in range(dim):
            v_time_0[:,d]  = gamma.rvs(a = 2, size = n, scale = 3)
            v_time_1[:, d] = gamma.rvs(a = 2, size = n, scale = 3)

            v_star_time_0[:,d] = gamma.rvs(a = 3, size = n, scale = 2)
            v_star_time_1[:,d] = gamma.rvs(a = 3, size=n, scale = 2)

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
        with open('data_file/high_dim/high_dim_gamma_d_'+str(dim) + "_num_" + str(run_number) + '.pkl', 'wb') as f:
            dill.dump(h_0, file=f)
            dill.dump(B, file=f)
            dill.dump(h_1, file=f)
            dill.dump(v_time_0, file=f)
            dill.dump(v_time_1, file=f)
            dill.dump(v_star_time_0, file=f)
            dill.dump(v_star_time_1, file=f)
            dill.dump(y_00_N, file=f)
            dill.dump(y_01_N, file=f)
            dill.dump(y_10_N, file=f)
            dill.dump(y_11_N_true, file=f)
            dill.dump(a, file=f)
            dill.dump(b, file=f)
            dill.dump(c, file=f)
            dill.dump(nn_index_array, file = f)
            f.close()


import joblib

result = joblib.Parallel(n_jobs = len(dim_array))(joblib.delayed(create_data)(dim) for dim in dim_array)


################