import numpy as np
import torch
from scipy.stats import gamma
import joblib
import dill
from script.library.OT_utils import *

np.random.seed(seed= 2023)


n = 10

m = 10
num_of_repeat = 10
dim_array = [50,100]
num_iter_array = [1,5,10,25,50,100,200]

sinkhorn_parameter = [10,30,90]
#num_iter_array = [1,2]

run_vector = ["full","CiC","maxSW_sampling","maxSW_adam","SW","sinkhorn"]

def run_one_experiment(dim):
    def run_one_data_set(run_number):
        with open('data_file/high_dim/high_dim_gamma_small_n_d_' + str(dim) + "_num_" + str(run_number) + '.pkl', 'rb') as f:
            h_0 = dill.load(file=f)
            B = dill.load(file = f)
            h_1 = dill.load(file=f)
            v_time_0 = dill.load(file=f)
            v_time_1 = dill.load(file=f)
            v_star_time_0 = dill.load(file=f)
            v_star_time_1 = dill.load(file=f)
            y_00_N = dill.load(file=f)
            y_01_N = dill.load(file=f)
            y_10_N = dill.load(file=f)
            y_11_N_true =dill.load(file=f)
            a = dill.load(file=f)
            b = dill.load(file=f)
            c = dill.load(file=f)
            nn_index_array = dill.load(file=f)
            f.close()
        time_dict = {}
        emd_dict = {}
        PCA_counterfactual, time_pca = PCA_sliced_OT_causal_estimate(a, b, c, nn_index_array=nn_index_array)
        emd_PCA = emd_dist(PCA_counterfactual, y_11_N_true.transpose())

        maxsw_sampling_counterfactual, time_maxsw_1 = MSW_by_sampling_causal_estimate(a, b, c,
                                                                                      nn_index_array=nn_index_array,
                                                                                      num_projs = 10)
        emd_maxsw_1 = emd_dist(maxsw_sampling_counterfactual, y_11_N_true.transpose())
        time_dict["PCA"] = time_pca
        emd_dict["PCA"] = emd_PCA

        time_dict["proposed"] = time_maxsw_1
        emd_dict["proposed"] = emd_maxsw_1

        return time_dict,emd_dict

    one_result = joblib.Parallel(n_jobs = m)(joblib.delayed(run_one_data_set)(run_num) for run_num in range(m))
    return one_result

result = {}
for dd in range(len(dim_array)):
    result[dim_array[dd]] = run_one_experiment(dim_array[dd])

with open('result_file/high_dim_gamma_small_n_PCA.pkl', 'wb') as f:
    dill.dump(result,file = f)
################
# calculate average emd and 2*sd:
d = 100
num = 10
time_vec = np.zeros((num))
emd_vec =  np.zeros((num))
for i in range(num):
    time_vec[i] = result[d][i][0]["PCA"]
    emd_vec[i] = result[d][i][1]["PCA"]
print(np.mean(emd_vec))
print(np.std(emd_vec))

time_vec_pp = np.zeros((num))
emd_vec_pp =  np.zeros((num))
for i in range(num):
    time_vec_pp[i] = result[d][i][0]["proposed"]
    emd_vec_pp[i] = result[d][i][1]["proposed"]
print(np.mean(emd_vec_pp))
print(np.std(emd_vec_pp))



#result[dim][number_of_dataset][time/emd]