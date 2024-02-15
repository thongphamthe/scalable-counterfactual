import numpy as np
import torch
from scipy.stats import gamma

from script.library.OT_utils import *

np.random.seed(seed= 2023)

n_array        = [5000,10000,15000,20000]
num_iter_array = [1,2,10,50,100,200,500]
m       = 10
alpha   = 0.5

w1 = 0.5; w2 = 0.5; w3 = 0.5; w4 = 0.5;
loc1 = 1; sd1 = 1; loc2 = 5; sd2 = 1;
loc3 = 5; sd3 = 1; loc4 = 1; sd4 = 1;
loc5 = 5; sd5 = 1; loc6 = 1; sd6 = 1;
loc7 = 1; sd7 = 1; loc8 = 5; sd8 = 1;


run_vector = ["full","CiC","maxSW_sampling","maxSW_adam","SW"]

#

def create_data(num_sample):
    import dill
    w1 = 0.5;
    w2 = 0.5;
    w3 = 0.5;
    w4 = 0.5;

    loc1 = 1;
    sd1 = 1;

    loc2 = 5;
    sd2 = 1;

    loc3 = 2;
    sd3 = 1;

    loc4 = 4;
    sd4 = 1;


    loc5 = 2;
    sd5 = 1;

    loc6 = 4;
    sd6 = 1;

    loc7 = 1;
    sd7 = 1;

    loc8 = 5;
    sd8 = 1;

    m = 10
    h_0 = np.array([[1, alpha], [alpha, 1]])
    h_1 = np.array([[1, -alpha], [-alpha, 1]])
    for run_number in range(m):
        v_time_0,v_time_1, v_star_time_0, v_star_time_1 =create_data_mixtureGauss(num_sample,w1,w2 ,w3,w4,loc1,sd1,loc2,sd2,loc3,sd3,loc4,
                                            sd4,loc5,sd5,loc6,sd6,loc7,sd7,loc8,sd8)
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


        with open('data_file/n_mixtureG/n_mixtureG_n_' + str(num_sample) + "_num_" + str(run_number) + '.pkl', 'wb') as f:
            dill.dump(h_0, file=f)
            dill.dump(h_1, file=f)
            dill.dump(v_time_0, file=f)
            dill.dump(v_time_1, file=f)
            dill.dump(v_star_time_0, file=f)
            dill.dump(v_star_time_1, file=f)
            dill.dump(m,file = f)
            dill.dump(alpha, file=f)
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

