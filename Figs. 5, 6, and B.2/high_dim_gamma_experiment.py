import numpy as np
import torch
from scipy.stats import gamma
import joblib
import dill
from script.library.OT_utils import *

np.random.seed(seed= 2023)


n = 5000

m = 10
num_of_repeat = 10
dim_array = [2,10,50,100]
num_iter_array = [1,5,10,25,50,100,200]

sinkhorn_parameter = [10,30,90]


run_vector = ["full","CiC","maxSW_sampling","maxSW_adam","SW","sinkhorn"]

def run_one_experiment(dim):
    def run_one_data_set(run_number):
        with open('data_file/high_dim/high_dim_gamma_d_' + str(dim) + "_num_" + str(run_number) + '.pkl', 'rb') as f:
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
        for index,value in enumerate(run_vector):
            if value == "full":
                full_OT_counterfactual, time_full = full_OT_causal_estimate(a, b, c, nn_index_array=nn_index_array,
                                                                         iter=10000000, core=1)
                emd_full = emd_dist(full_OT_counterfactual,y_11_N_true.transpose())
                if value not in time_dict.keys():
                    time_dict[value] = []
                    emd_dict[value] = []
                time_dict[value].append(time_full)
                emd_dict[value].append(emd_full)
            elif value == "CiC":
                if value not in time_dict.keys():
                    time_dict[value] = []
                    emd_dict[value] = []
                CiC_counterfactual, time_cic = marginal_OT_causal_estimate(a, b, c, nn_index_array=nn_index_array)
                emd_cic =  emd_dist(CiC_counterfactual,y_11_N_true.transpose())
                time_dict[value].append(time_cic)
                emd_dict[value].append(emd_cic)
            elif value == "maxSW_sampling":
                if value not in time_dict.keys():
                    time_dict[value] = {}
                    emd_dict[value] = {}

                for num_proj in num_iter_array:
                    if num_proj not in time_dict[value].keys():
                        time_dict[value][num_proj] = []
                        emd_dict[value][num_proj] = []
                    for kk in range(num_of_repeat):
                        maxsw_sampling_counterfactual, time_maxsw_1 = MSW_by_sampling_causal_estimate(a, b, c,
                                                                                                nn_index_array=nn_index_array,
                                                                                            num_projs = num_proj)
                        emd_maxsw_1 = emd_dist(maxsw_sampling_counterfactual, y_11_N_true.transpose())

                        time_dict[value][num_proj].append(time_maxsw_1)
                        emd_dict[value][num_proj].append(emd_maxsw_1)
            elif value == "maxSW_adam":
                if value not in time_dict.keys():
                    time_dict[value] = {}
                    emd_dict[value] = {}

                for num_proj in num_iter_array:
                    if num_proj not in time_dict[value].keys():
                        time_dict[value][num_proj] = []
                        emd_dict[value][num_proj] = []
                    for kk in range(num_of_repeat):
                        maxsw_optimization_counterfactual, time_maxsw_2 = MSW_by_optimization_causal_estimate(a, b, c,
                                                                                                                nn_index_array=nn_index_array,
                                                                                                                num_iter=num_proj)
                        emd_maxsw_2 = emd_dist(maxsw_optimization_counterfactual, y_11_N_true.transpose())

                        time_dict[value][num_proj].append(time_maxsw_2)
                        emd_dict[value][num_proj].append(emd_maxsw_2)
            elif value == "SW":
                if value not in time_dict.keys():
                    time_dict[value] = {}
                    emd_dict[value] = {}

                for num_proj in num_iter_array:
                    if num_proj not in time_dict[value].keys():
                        time_dict[value][num_proj] = []
                        emd_dict[value][num_proj] = []
                    for kk in range(num_of_repeat):
                        _, sw_cf, time_sw = sliced_OT_causal_estimate(a, b, c, nn_index_array=nn_index_array, num_projs = num_proj)
                        emd_sw = emd_dist(sw_cf, y_11_N_true.transpose())
                        time_dict[value][num_proj].append(time_sw)
                        emd_dict[value][num_proj].append(emd_sw)
            elif value == "sinkhorn":
                if value not in time_dict.keys():
                    time_dict[value] = {}
                    emd_dict[value] = {}
                for lamb in sinkhorn_parameter:
                    if lamb not in time_dict[value].keys():
                        time_dict[value][lamb] = []
                        emd_dict[value][lamb] = []

                    sinkhorn_cf, time_sinkhorn,_ = Sinkhorn_causal_estimate(a, b, c, nn_index_array=nn_index_array,
                                                                    reg_e = lamb,tol = pow(10,-2),method = "sinkhorn_log")
                    emd_sinkhorn = emd_dist(sinkhorn_cf, y_11_N_true.transpose())
                    time_dict[value][lamb].append(time_sinkhorn)
                    emd_dict[value][lamb].append(emd_sinkhorn)

        return time_dict,emd_dict

    one_result = joblib.Parallel(n_jobs = m)(joblib.delayed(run_one_data_set)(run_num) for run_num in range(m))
    return one_result

result = {}
for dd in range(len(dim_array)):
    result[dim_array[dd]] = run_one_experiment(dim_array[dd])

with open('result_file/high_dim_gamma.pkl', 'wb') as f:
    dill.dump(result,file = f)
################

#result[dim][number_of_dataset][time/emd]
