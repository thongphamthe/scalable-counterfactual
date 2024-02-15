import numpy as np
import torch
from scipy.stats import gamma

from script.library.OT_utils import *

np.random.seed(seed= 2023)

n_array        = [5000,10000,15000,20000]
num_iter_array = [1,5,10,25,50,100,200] #for SW-group
mm       = 10
alpha   = 0.5
sinkhorn_parameter = [10,30,90]


run_vector = ["full","CiC","maxSW_sampling","maxSW_adam","SW","sinkhorn"]

#

def run_one_experiment(num_sample):
    mm = 10
    import dill
    number_of_row = len(run_vector)
    if "maxSW_sampling" in run_vector:
        number_of_row += len(num_iter_array) - 1
    if "maxSW_adam" in run_vector:
        number_of_row += len(num_iter_array) - 1
    if "SW" in run_vector:
        number_of_row += len(num_iter_array) - 1
    if "sinkhorn" in run_vector:
        number_of_row += len(sinkhorn_parameter) - 1
    result_one = np.full((mm, number_of_row, 2), -1.00)
    for run_number in range(mm):
        with open('data_file/n_gamma/n_gamma_n_' + str(num_sample) + "_num_" + str(run_number) + '.pkl', 'rb') as f:
            h_0 = dill.load(file=f)
            h_1 = dill.load(file=f)
            v_time_0 = dill.load(file=f)
            v_time_1 = dill.load(file=f)
            v_star_time_0 = dill.load(file=f)
            v_star_time_1 = dill.load(file=f)

            m = dill.load(file=f)
            alpha = dill.load(file=f)
            a1 = dill.load(file=f)
            scale1 = dill.load(file=f)
            a2 = dill.load(file=f)
            scale2 = dill.load(file=f)
            a3 = dill.load(file=f)
            scale3 = dill.load(file=f)
            a4 = dill.load(file=f)
            scale4 = dill.load(file=f)
            y_00_N = dill.load(file=f)
            y_01_N = dill.load(file=f)
            y_10_N = dill.load(file=f)
            y_11_N_true = dill.load(file=f)
            a = dill.load(file=f)
            b = dill.load(file=f)
            c = dill.load(file=f)
            nn_index_array = dill.load(file=f)
            f.close()
        result_one[run_number, :] = one_experiment(a, b, c, y_11_N_true, nn_index_array,
                                          run_experiments=run_vector, num_proj=num_iter_array,
                                          sinkhorn_parameter = sinkhorn_parameter)
        print("n_samples: " + str(num_sample) + "- run: " + str(run_number))
    return result_one


import joblib
result = joblib.Parallel(n_jobs=len(n_array))(joblib.delayed(run_one_experiment)(n_sample) for n_sample in n_array)

import dill
with open('result_file/n_gamma_result.pkl', 'wb') as f:
    dill.dump(result,file = f)
