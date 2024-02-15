import numpy as np
import torch
from scipy.spatial.distance import cdist
from script.library.maxsw_utils import *

def marginal_1D_OT(source,target):
    import time
    #transported_values = torch.zeros_like(source)
    dim_k = source.shape[1]

    start = time.perf_counter_ns()
    #print(source)
    #print(target)
    source_sorted_index = torch.argsort(source, dim = 0)
    #print(source_sorted_index)
    target_sorted_index = torch.argsort(target, dim = 0)
    #print(target_sorted_index)
    new_transported_values = torch.zeros_like(source)
    for k in range(dim_k):
        #print(source_sorted_index[:,k])
        #print(target_sorted_index[:,k])
        new_transported_values[source_sorted_index[:,k],k] = target[target_sorted_index[:,k],k]

    #transported_values[source_sorted_index] = target[target_sorted_index]
    #print(transported_values)
    #print(new_transported_values)
    end = time.perf_counter_ns()

    total_time = (end - start)
    #print(total_time)
    return new_transported_values, total_time/1000000.0 # time in miliseconds


def marginal_1D_OT_old(source,target):
    import time
    transported_values = torch.zeros_like(source)
    dim_k = source.shape[1]
    total_time = 0.0
    print(source)
    print(target)
    for k in range(dim_k):
        start = time.perf_counter_ns()
        proj_source_sorted = torch.argsort(source[:,k])
        proj_target_sorted = torch.argsort(target[:,k])
        print(proj_source_sorted)
        print(proj_target_sorted)
        transported_values[proj_source_sorted,k] = target[proj_target_sorted,k]
        end = time.perf_counter_ns()
        total_time += (end - start)
    #print(total_time)
    print(transported_values)
    return transported_values, total_time/1000000.0 # time in miliseconds

def full_OT(U,V,iter = 100000000, core = 1):
    # compute OT between control pre and control_post with uniform marginals
    import time
    #print(cost_matrix.shape)
    start_time = time.perf_counter_ns()
    cost_matrix = cdist(U, V, 'sqeuclidean')
    control_ot_map = ot.emd(np.ones(U.shape[0])/U.shape[0],np.ones(V.shape[0])/V.shape[0],cost_matrix,numItermax = iter, numThreads = core)
    transported_U = np.matmul(control_ot_map,V) * U.shape[0]
    end_time = time.perf_counter_ns()
    elapsed = end_time - start_time
    return control_ot_map, transported_U, elapsed/1000000.0


def sliced_OT(source,target,num_projs = 100):
    #source, target: torch
    import time

    sliced_values = torch.zeros_like(source)
    dim_k = source.shape[1]
    start = time.perf_counter_ns()

    U_matrix = torch.randn(num_projs, dim_k)
    #print(U_matrix)
    U_matrix = torch.nn.functional.normalize(U_matrix,dim = 1)
    #print(U_matrix)
    proj_source = torch.inner(source, U_matrix)
    proj_target = torch.inner(target, U_matrix)
    #print(proj_source)
    #print(proj_target)
    proj_source_indices = torch.argsort(proj_source, dim=0)
    # print(proj_source_sorted.shape)
    #print(proj_source_indices)
    proj_target_indices = torch.argsort(proj_target, dim=0)
    #print(proj_target_indices)

    for k in range(num_projs):
        sliced_values[proj_source_indices[:,k],:] += target[proj_target_indices[:,k],:]

    #print(sliced_values.shape)
    sliced_values /= num_projs
    end = time.perf_counter_ns()
    total_time = (end - start)
    #print(sliced_values)
    return sliced_values, total_time/1000000.0

def sliced_OT_old(source,target,num_projs = 100):
    #source, target: torch
    import time
    import torch
    sliced_values = torch.zeros_like(source)
    dim_k = source.shape[1]
    print(dim_k)
    total_time = 0.0
    for k in range(num_projs):
        start = time.perf_counter_ns()
        u = torch.randn(dim_k)
        print(u)
        print("After nomalized:")
        u /= torch.linalg.vector_norm(u)

        print(u)
        proj_source = torch.inner(source,u)
        proj_target = torch.inner(target,u)

        proj_source_sorted = torch.argsort(proj_source)
        proj_target_sorted = torch.argsort(proj_target)
        sliced_values[proj_source_sorted,:] += target[proj_target_sorted,:]
        end = time.perf_counter_ns()
        total_time += (end - start)
    #print(sliced_values.shape)
    sliced_values /= num_projs
    print(sliced_values)
    return sliced_values, total_time/1000000.0


def maxsw_by_sampling_OT(source,target,num_projs = 100, cores = 1):
    import time
    import torch
    sliced_values = torch.zeros_like(source)
    dim_k = source.shape[1]


    total_time = 0.0

    start = time.perf_counter_ns()

    U_matrix = torch.randn(num_projs,dim_k)
    U_matrix = torch.nn.functional.normalize(U_matrix, dim=1)
    proj_source = torch.inner(source, U_matrix)
    proj_target = torch.inner(target, U_matrix)
    #print(proj_source.shape)
    proj_source_sorted,proj_source_indices = torch.sort(proj_source, dim = 0)
    #print(proj_source_sorted.shape)
    #print(proj_source_sorted)
    proj_target_sorted,proj_target_indices = torch.sort(proj_target, dim = 0)
    #distance = 0
    #print(proj_target)
    #print(proj_target_sorted)
    #uu = proj_target[proj_target_sorted]
    #print(uu)
    #print(uu.shape)
    all_dist = torch.sum(torch.abs(proj_target_sorted - proj_source_sorted),dim = 0)
    #print(all_dist)
    best_index = torch.argmax(all_dist)

    sliced_values[proj_source_indices[:,best_index],:] = target[proj_target_indices[:,best_index],:]
    end = time.perf_counter_ns()

    total_time += (end - start)

    return U_matrix[best_index,:],all_dist[best_index],sliced_values,total_time/1000000.0

def single_sliced_OT(source,target,project_vector):
    import time
    import torch
    sliced_values = torch.zeros_like(source)
    dim_k = source.shape[1]


    total_time = 0.0

    start = time.perf_counter_ns()

    proj_source = torch.inner(source, project_vector)
    proj_target = torch.inner(target, project_vector)
    #print(proj_source.shape)
    proj_source_sorted,proj_source_indices = torch.sort(proj_source)
    #print(proj_source_sorted.shape)
    #print(proj_source_sorted)
    proj_target_sorted,proj_target_indices = torch.sort(proj_target)
    #distance = 0
    #print(proj_target)
    #print(proj_target_sorted)
    #uu = proj_target[proj_target_sorted]
    #print(uu)
    #print(uu.shape)

    sliced_values[proj_source_indices,:] = target[proj_target_indices,:]
    end = time.perf_counter_ns()

    total_time += (end - start)

    return sliced_values,total_time/1000000.0

def max_sliced_OT(source,target,num_iter = 100, solver = "Adam"):
    import time
    sliced_values = torch.zeros_like(source)
    dim_k = source.shape[1]

    slicer = Slicer(d = dim_k)
    if solver == "Adam":
        slicer_optimizer = torch.optim.Adam(params = slicer.parameters())
    else:
        print("Not yet implemented solver!")
        exit()

    project_vec,proj_source_sorted_final,proj_target_sorted_final,max_dist, total_time = MaxSW(source, target, slicer,slicer_optimizer,num_iter)
    #print(proj_source_sorted_final)
    #print(proj_target_sorted_final)
    start = time.perf_counter_ns()
    sliced_values[proj_source_sorted_final,:] = target[proj_target_sorted_final,:]
    end = time.perf_counter_ns()
    total_time += (end - start)
    return project_vec,max_dist,sliced_values,total_time/1000000.0


def Sinkhorn_OT(source,target,iter = 100000, reg_e = 1, tol = pow(10,-5), method = "sinkhorn"):
    import time
    import ot
    total_time = 0
    # convert source and target from torch to numpy array
    source_np = source.detach().numpy()
    target_np = target.detach().numpy()
    start = time.perf_counter_ns()
    #print(source_np.shape)
    #print(target_np.shape)
    ot_sinkhorn = ot.da.SinkhornTransport(max_iter = iter,reg_e = reg_e,
                                          log = True, tol = tol, method = method)
    ot_sinkhorn.fit(Xs=source_np, Xt=target_np)
    transported_U = ot_sinkhorn.transform(Xs = source_np)
    end = time.perf_counter_ns()
    total_time += (end - start)
    #print(transported_U.shape)
    transported_U_torch = torch.from_numpy(transported_U)
    transported_U_torch = transported_U_torch.to(torch.float)
    return ot_sinkhorn.coupling_,transported_U_torch,total_time/1000000.0,ot_sinkhorn


def run_ot(U, V, standardize=False):
    N = U.shape[0]
    M = V.shape[0]

    if standardize:
        new_U = (U - np.mean(U, axis=0)) / np.std(U, axis=0)
        new_V = (V - np.mean(V, axis=0)) / np.std(V, axis=0)
        U, V = new_U, new_V

    pre_marginal = np.ones((N)) / N
    post_marginal = np.ones((M)) / M
    cost_matrix = cdist(U, V, 'sqeuclidean')

    ot_plan = ot.emd(pre_marginal, post_marginal, cost_matrix)

    return ot_plan

def nearest_neighbor_index_finding(W,U):
    #print(W.shape)
    index_array = np.zeros(W.shape[0])
    for w_index, unit in enumerate(W):
        absolute_differences = torch.subtract(U,unit)
        distances = torch.norm(absolute_differences, dim = -1)

        closest_index = np.argsort(distances)[:1]
        index_array[w_index] = closest_index
    return index_array

def nn_interpolation(W_shape,transported_U,index_array):
    #counterfactual_outcomes = torch.empty(W_shape)
    counterfactual_outcomes = transported_U[index_array,:]
    return counterfactual_outcomes

def CiC_mapping(control_pre, treatment_pre, transported_control):
    #control_pre : n x d, treatment_pre: n x d
    # for each row of treatment_pre: substract the matrix control_pre by that row
    # then look for the index of the biggest value that not
    counterfactual_outcomes = torch.empty(treatment_pre.shape)
    #sorted_control_pre = torch.sort(control_pre,dim = 0)
    for i in range(treatment_pre.shape[0]):
        for j in range(treatment_pre.shape[1]):
            temp = [value.item() - treatment_pre[i,j].item() for value in control_pre[:,j]]
            #print(temp)
            temp = np.array(temp)
            #print(temp)
            choose_index = np.where(temp <= 0)
            choose_index = np.array(choose_index).flatten()

            if len(choose_index) > 0:
                vec_temp = temp[temp <= 0]
                min_index = np.argwhere(vec_temp == np.amax(vec_temp))
            else:
                choose_index = np.where(temp >= 0)
                choose_index = np.array(choose_index).flatten()
                vec_temp = temp[temp >= 0]
                min_index = np.argwhere(vec_temp == np.amin(vec_temp))
            #print(min_index.flatten())
            #print(greater_than_zero_index)
            min_index = np.array(min_index.flatten(), dtype = "int32")

            min_index = choose_index[min_index]
            #print(transported_control.shape)
            if len(min_index) > 1:
            #print(transported_control[min_index, j])
            #print(transported_control[min_index,j].detach().numpy)
                cf_value = torch.amax(transported_control[min_index,j])
            else:
                cf_value = transported_control[min_index, j].item()
            #print(vec_temp)
            #print(min_index)
            #print(transported_control[min_index,j].clone().detach())
            #print(cf_value)
            counterfactual_outcomes[i,j] = cf_value
    return counterfactual_outcomes

def nearest_neighbor_interpolation_old(W,U,transported_U, base_vector = None, num_of_point = 1):
    # calculate counterfactual treatments by finding nearest control and transporting
    counterfactual_outcomes = torch.empty(W.shape)
    W_project = W
    U_project = U
    transported_U_project = transported_U
    #print("Start here")
    #print(transported_U.shape)
    if base_vector is not None:
        W_project = torch.inner(W,base_vector)
        U_project = torch.inner(U,base_vector)
        transported_U_project = torch.inner(transported_U,base_vector)
    #print(transported_U_project.shape)
    #print(W.shape)
    #print(transported_U.shape)
    for w_index, unit in enumerate(W_project):
        absolute_differences = torch.subtract(U_project,unit)
        distances = torch.norm(absolute_differences, dim = -1)
        #print(distances.shape)
        closest_index = np.argsort(distances)[:num_of_point]
        #print(closest_index)
        #print(transported_U[closest_index,:])
        #print(np.mean(transported_U[closest_index,:],axis = 0))
        if base_vector is not None:
            counterfactual_outcomes[w_index, :] = torch.mean(transported_U_project[closest_index], 0)
        else:
            counterfactual_outcomes[w_index,:] = torch.mean(transported_U_project[closest_index,:],0)

    return counterfactual_outcomes

def marginal_OT_causal_estimate(y_00_n, y_01_n,y_10_n,nn_index_array):
    transported_U, time = marginal_1D_OT(y_00_n,y_01_n)

    treatment_counterfactuals = CiC_mapping(y_00_n,y_10_n,transported_U)
    return treatment_counterfactuals, time

def full_OT_causal_estimate(y_00_n, y_01_n,y_10_n,nn_index_array, iter = 100000000, core = 1):
    control_ot_map, transported_U, time = full_OT(y_00_n,y_01_n, iter = iter, core=core)
    treatment_counterfactuals_ot = nn_interpolation(y_10_n.shape,transported_U,nn_index_array)
    return treatment_counterfactuals_ot, time

def Sinkhorn_causal_estimate(y_00_n, y_01_n,y_10_n,nn_index_array,
                             iter = 100000,reg_e = 1, tol = pow(10,-3), method = "sinkhorn_log"):
    control_ot_map, transported_U, time, ot_sinkhorn = Sinkhorn_OT(y_00_n,y_01_n, iter = iter, reg_e = reg_e,tol = tol,method = method)
    #print(transported_U)
    treatment_np = y_10_n.detach().numpy()
    treatment_counterfactuals_transformed =  ot_sinkhorn.transform(treatment_np)
    treatment_counterfactuals_ot = nn_interpolation(y_10_n.shape,transported_U,nn_index_array)
    return treatment_counterfactuals_ot, time, treatment_counterfactuals_transformed

def sliced_OT_causal_estimate(y_00_n, y_01_n,y_10_n, nn_index_array, num_projs = 100, num_of_points = 1):
    sliced_OT_value, time = sliced_OT(y_00_n,y_01_n, num_projs = num_projs)
    treatment_counterfactuals_sliced = nn_interpolation(y_10_n.shape,sliced_OT_value,nn_index_array)
    return sliced_OT_value,treatment_counterfactuals_sliced,time

def MSW_by_sampling_causal_estimate(y_00_n, y_01_n,y_10_n, nn_index_array, num_projs = 100, num_of_points = 1):
    best_vector,best_dist,sliced_OT_value, time = maxsw_by_sampling_OT(y_00_n,y_01_n, num_projs = num_projs)

    treatment_counterfactuals = nn_interpolation(y_10_n.shape,sliced_OT_value,nn_index_array)
    #print(best_dist)
    return treatment_counterfactuals,time


def MSW_by_sampling_causal_estimate_for_experiments(y_00_n, y_01_n,y_10_n, nn_index_array, num_proj_array):
    best_vector,best_dist,sliced_OT_value, time = maxsw_by_sampling_OT_increaments(y_00_n,y_01_n, num_proj_array)

    treatment_counterfactuals = nn_interpolation(y_10_n.shape,sliced_OT_value,nn_index_array)
    #print(best_dist)
    return treatment_counterfactuals,time
def maxsw_by_sampling_OT_increaments(source,target,num_projs_array):
    # num_projs_array: an increasing list of number of projections

    import time
    import torch
    sliced_values = torch.zeros_like(source)
    dim_k = source.shape[1]


    total_time = 0.0

    start = time.perf_counter_ns()

    U_matrix = torch.randn(num_projs,dim_k)
    U_matrix = torch.nn.functional.normalize(U_matrix, dim=1)
    proj_source = torch.inner(source, U_matrix)
    proj_target = torch.inner(target, U_matrix)
    #print(proj_source.shape)
    proj_source_sorted,proj_source_indices = torch.sort(proj_source, dim = 0)
    #print(proj_source_sorted.shape)
    #print(proj_source_sorted)
    proj_target_sorted,proj_target_indices = torch.sort(proj_target, dim = 0)
    #distance = 0
    #print(proj_target)
    #print(proj_target_sorted)
    #uu = proj_target[proj_target_sorted]
    #print(uu)
    #print(uu.shape)
    all_dist = torch.sum(torch.abs(proj_target_sorted - proj_source_sorted),dim = 0)
    #print(all_dist)
    best_index = torch.argmax(all_dist)

    sliced_values[proj_source_indices[:,best_index],:] = target[proj_target_indices[:,best_index],:]
    end = time.perf_counter_ns()

    total_time += (end - start)

    return U_matrix[best_index,:],all_dist[best_index],sliced_values,total_time/1000000.0


def single_sliced_causal_estimate(y_00_n, y_01_n,y_10_n, nn_index_array, project_vector):
    sliced_OT_value, time = single_sliced_OT(y_00_n,y_01_n, project_vector)

    treatment_counterfactuals = nn_interpolation(y_10_n.shape,sliced_OT_value,nn_index_array)
    #print(best_dist)
    return treatment_counterfactuals,time


def MSW_by_optimization_causal_estimate(y_00_n, y_01_n,y_10_n, nn_index_array, num_iter = 1000, num_of_points = 1):
    _,_,sliced_OT_value,time = max_sliced_OT(y_00_n,y_01_n, num_iter=num_iter)
    treatment_counterfactuals = nn_interpolation(y_10_n.shape,sliced_OT_value,nn_index_array)
    return treatment_counterfactuals, time

#this is the optimization



def plot_density(sample_0,sample_1,title,
                 xmin , xmax ,
                 ymin , ymax ,
                 x_size = None, y_size = None,
                 xticks = None, yticks = None, tight = True):
    import numpy as np
    import scipy.stats as st
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    #plt.tight_layout()
    import scienceplots
    plt.style.use(['science'])
    with plt.style.context(['science', 'ieee']):
        #plt.tight_layout()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
    #values = np.vstack([y_11_N_true[0], y_11_N_true[1]])
        values = np.vstack([sample_0, sample_1])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        fig = plt.figure()
        ax = fig.gca()


    # Contourf plot
        cfset = ax.contourf(xx, yy, f, cmap='Greys')
        #ax.set_xlabel(r'$Y_0$',fontsize = x_size)
        #ax.set_ylabel(r'$Y_1$', fontsize = y_size)
        ax.set_xlabel("First dimension", fontsize=x_size)
        ax.set_ylabel("Second dimension", fontsize = y_size)
        plt.title(title)
        if x_size is not None:
            xticklabels = ax.get_xticklabels()
            yticklabels = ax.get_yticklabels()
            ax.set_xticklabels(xticklabels, fontsize=x_size)
            ax.set_yticklabels(yticklabels, fontsize=y_size)

        if xticks is not None:
            plt.xticks(ticks = xticks,labels=xticks)
        if yticks is not None:
            plt.yticks(ticks=yticks, labels=yticks)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        #plt.subplots_adjust(left=2, right=3, bottom=2, top=3)
        ax.set_aspect("equal")
        if tight:
            plt.tight_layout()
        plt.show()


def plot_density_ax(ax,sample_0,sample_1,title,xtitle, ytitle,
                 xmin , xmax ,
                 ymin , ymax ,
                 x_size = None, y_size = None,
                 xticks = None, yticks = None, tight = True,xlabel="",ylabel =""):
    import numpy as np
    import scipy.stats as st
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    #plt.tight_layout()
    import scienceplots
    with plt.style.context(['science', 'ieee']):
        #plt.tight_layout()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
    #values = np.vstack([y_11_N_true[0], y_11_N_true[1]])
        values = np.vstack([sample_0, sample_1])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        #fig = plt.figure()
        #ax = fig.gca()


    # Contourf plot
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        #ax.set_xlabel(r'$Y_0$',fontsize = x_size)
        #ax.set_ylabel(r'$Y_1$', fontsize = y_size)
        ax.set_xlabel(xlabel, fontsize = x_size)
        ax.set_ylabel(ylabel, fontsize = y_size)
        #ax.title.set_text(title)
        ax.text(xtitle,ytitle,title,fontsize = x_size,transform=ax.transAxes,ha = "left",verticalalignment='center')
        if x_size is not None:
            xticklabels = ax.get_xticklabels()
            ax.set_xticklabels(xticklabels, fontsize=x_size)
        #if xticks is not None:
            plt.xticks(ticks = xticks,labels=xticks)
        #if yticks is not None:
        else:
            xticklabels = ax.get_xticklabels()
            ax.set_xticklabels(labels= "")
        if y_size is not None:
            yticklabels = ax.get_yticklabels()
            ax.set_yticklabels(yticklabels, fontsize=y_size)
            plt.yticks(ticks=yticks, labels=yticks)
        else:
            yticklabels = ax.get_yticklabels()
            ax.set_yticklabels(labels = "")
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        #plt.subplots_adjust(left=2, right=3, bottom=2, top=3)
        ax.set_aspect("equal")


def emd_dist(dist1, dist2, iter = 100000000):
    # standard OT dist between two discrete distributions
    cost_matrix = cdist(dist1, dist2, 'sqeuclidean')
    return ot.emd2(np.ones(dist1.shape[0])/dist1.shape[0],np.ones(dist2.shape[0])/dist2.shape[0],cost_matrix,numItermax = iter)

def one_experiment(a,b,c,y_11_N_true,nn_index_array,
                   run_experiments = ["full","CiC","maxSW_sampling","maxSW_adam","SW","sinkhorn"],
                   num_proj = [100],
                   sinkhorn_parameter = [0.1]):
    number_of_row = len(run_experiments)
    if "maxSW_sampling" in run_experiments:
        number_of_row += len(num_proj) - 1
    if "maxSW_adam" in run_experiments:
        number_of_row += len(num_proj) - 1
    if "SW" in run_experiments:
        number_of_row += len(num_proj) - 1
    if "sinkhorn" in run_experiments:
        number_of_row += len(sinkhorn_parameter) - 1
    result = np.full((number_of_row,2),-1.0)
    col_counter = -1
    for i,value in enumerate(run_experiments):
        #print(value)
        if value == "CiC":
            CiC_counterfactual, time_1 = marginal_OT_causal_estimate(a, b, c,nn_index_array = nn_index_array)

            col_counter += 1
            result[col_counter,0] = time_1
            result[col_counter,1] = emd_dist(CiC_counterfactual, y_11_N_true.transpose())
        elif value == "full":
            full_OT_counterfactual, time_2 = full_OT_causal_estimate(a, b, c, nn_index_array = nn_index_array, iter=100000000, core=1)
            col_counter += 1
            result[col_counter,0] = time_2
            result[col_counter,1] = emd_dist(full_OT_counterfactual, y_11_N_true.transpose())
        elif value == "maxSW_sampling":
            for j,val_j in enumerate(num_proj):
                maxsw_sampling_counterfactual, time_3 = MSW_by_sampling_causal_estimate(a, b, c, nn_index_array = nn_index_array, num_projs = val_j)

                col_counter += 1
                result[col_counter,0] = time_3

                result[col_counter,1] =  emd_dist(maxsw_sampling_counterfactual, y_11_N_true.transpose())
        elif value == "maxSW_adam":
            for j,val_j in enumerate(num_proj):
                maxsw_optimization_counterfactual, time_4 = MSW_by_optimization_causal_estimate(a, b, c,nn_index_array = nn_index_array, num_iter = val_j)

                col_counter += 1
                result[col_counter,0] = time_4
                result[col_counter,1] =  emd_dist(maxsw_optimization_counterfactual, y_11_N_true.transpose())
        elif value == "SW":
            for j,val_j in enumerate(num_proj):
                _, sw_cf, time_5 = sliced_OT_causal_estimate(a, b, c,nn_index_array = nn_index_array, num_projs = val_j)

                col_counter += 1
                result[col_counter,0] = time_5
                result[col_counter,1] = emd_dist(sw_cf, y_11_N_true.transpose())
        elif value == "sinkhorn":
            for j,val_j in enumerate(sinkhorn_parameter):
                sinkhorn_cf, time_sinkhorn, _ = Sinkhorn_causal_estimate(a, b, c, nn_index_array=nn_index_array,
                                                                         reg_e= val_j, tol=pow(10, -2),
                                                                         method="sinkhorn_log")
                col_counter += 1
                result[col_counter,0] = time_sinkhorn
                result[col_counter,1] = emd_dist(sinkhorn_cf, y_11_N_true.transpose())
        else:
            print("Not implemented method")
    return result



def get_cdf(outcomes):
    print(outcomes)
    # calculate the eCDF of a list of data
    # adapted from: https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
    sorted_outcomes = sorted(outcomes)
    values, counts = np.unique(sorted_outcomes,return_counts=True) # get unique values and their occurences
    sums = np.cumsum(counts) # get cumulative count up to each value and rescale for quantile
    quantiles = sums / sums[-1]
    return values, quantiles

def closest(list_, value, left=True):
  # helper function to get cdf values
  # given a list and a value, returns the closest element in the list (less/greater) than or equal to the value parameter
    if left:
        masked = [i if i <= value else -np.inf for i in list_]
        return max(masked)
    else:
        masked = [i if i >= value else np.inf for i in list_]
        return min(masked)

def nan_cic(treatment_pre,control_pre,control_post,verbose=False):
    # calculate the changes-in-changes estimator for both control and treatment group
    # return NaN if prediction if unit is outside support of other treatment arm

    # get the empirical CDFs of all observed groups
    control_pre_cdf = get_cdf(control_pre)
    control_post_cdf = get_cdf(control_post)
    treatment_pre_cdf = get_cdf(treatment_pre)

    # For each control pre-intervention, get the outcome quantile if it were a treated pre-intervention
    treatment_cf_pre_matches = [closest(control_pre_cdf[0],value) for value in treatment_pre]
    treatment_cf_pre_quantiles = [control_pre_cdf[1][np.where(control_pre_cdf[0]==new_value)[0][0]] if np.isfinite(new_value) else np.nan for new_value in treatment_cf_pre_matches]
    # Determine the post-intervention treatment value associated with each quantile found above
    treatment_cf_post_quantiles = [closest(control_post_cdf[1],quantile,left=False) for quantile in treatment_cf_pre_quantiles]
    treatment_cf = [control_post_cdf[0][np.where(control_post_cdf[1]==new_quantile)[0][0]] if np.isfinite(new_quantile) else np.nan for new_quantile in treatment_cf_post_quantiles]

    return treatment_cf

def full_bivariate_cic(treatment_pre,control_pre,control_post):
    # get coordinate-wise CiC estimates
    first_coordinate_counterfactuals = nan_cic(treatment_pre[:,0],control_pre[:,0],control_post[:,0])
    second_coordinate_counterfactuals = nan_cic(treatment_pre[:,1],control_pre[:,1],control_post[:,1])
    cic_results = np.vstack((first_coordinate_counterfactuals,second_coordinate_counterfactuals)).T
    return cic_results

def create_data_gamma(n, a1 = 1, scale1 = 1, a2 = 1, scale2 = 1,
                      a3 = 1, scale3 = 1, a4 = 1, scale4 = 1):
    from scipy.stats import gamma
    # time 0
    v_1 = gamma.rvs(a = a1, size=n, scale=scale1)
    v_2 = gamma.rvs(a = a2, size=n, scale=scale2)
    v_time_0 = np.array([v_1, v_2]).transpose()

    # time 1
    v_1 = gamma.rvs(a = a1, size=n, scale=scale1)
    v_2 = gamma.rvs(a = a2, size=n, scale=scale2)
    v_time_1 = np.array([v_1, v_2]).transpose()

    # treatment group's hidden variable time = 0
    v_star_1 = gamma.rvs(a = a3, size=n, scale = scale3)
    v_star_2 = gamma.rvs(a = a4, size=n, scale = scale4)
    v_star_time_0 = np.array([v_star_1, v_star_2]).transpose()

    # treatment group's hidden variable time = 1
    v_star_1 = gamma.rvs(a=a3, size=n, scale=scale3)
    v_star_2 = gamma.rvs(a=a4, size=n, scale=scale4)
    v_star_time_1 = np.array([v_star_1, v_star_2]).transpose()

    return v_time_0,v_time_1, v_star_time_0,v_star_time_1

def create_data_mixtureGauss(n, weight1 = 0.5, weight2 = 0.5,
                             weight3 = 0.5, weight4 = 0.5,
                             loc1 = 1, sd1 = 1, loc2 = 1, sd2 = 1,
                             loc3 = 1, sd3 = 1, loc4 = 1, sd4 = 1,
                             loc5 = 1, sd5 = 1, loc6 = 1, sd6 = 1,
                             loc7 = 1, sd7 = 1, loc8 = 1, sd8 = 1):
    from scipy.stats import norm as gauss
    from scipy.stats import bernoulli as bernoulli
    def mixture(n, loc1=1, sd1=1, loc2=5, sd2=1, weight=0.5):
        output = np.zeros(n)
        index = bernoulli.rvs(p=weight, size=n)
        for i, value in enumerate(index):
            if value >= 0.5:
                output[i] = gauss.rvs(loc=loc1, scale=sd1)
            else:
                output[i] = gauss.rvs(loc=loc2, scale=sd2)
        return output

    # control group's hidden variable at time t = 0
    v_1 = mixture(n=n, loc1 = loc1, sd1= sd1, loc2=loc2, sd2=sd2, weight = weight1)
    v_2 = mixture(n=n, loc1 = loc3, sd1= sd3 , loc2 = loc4, sd2=sd4, weight= weight2)
    v_time_0 = np.array([v_1, v_2]).transpose()

    # control group's hidden variable at time t = 1
    v_1 = mixture(n=n, loc1=loc1, sd1=sd1, loc2=loc2, sd2=sd2, weight=weight1)
    v_2 = mixture(n=n, loc1=loc3, sd1=sd3, loc2=loc4, sd2=sd4, weight=weight2)
    v_time_1 = np.array([v_1, v_2]).transpose()

    # treatment group's hidden variable at time t = 0
    v_star_1 = mixture(n=n, loc1 = loc5, sd1 = sd5, loc2 = loc6, sd2 = sd6, weight = weight3)
    v_star_2 = mixture(n=n, loc1 = loc7, sd1 = sd7, loc2 = loc8, sd2 = sd8, weight = weight4)
    v_star_time_0 = np.array([v_star_1, v_star_2]).transpose()

    # treatment group's hidden variable at time t = 0
    v_star_1 = mixture(n=n, loc1=loc5, sd1=sd5, loc2=loc6, sd2=sd6, weight=weight3)
    v_star_2 = mixture(n=n, loc1=loc7, sd1=sd7, loc2=loc8, sd2=sd8, weight=weight4)
    v_star_time_1 = np.array([v_star_1, v_star_2]).transpose()

    return v_time_0,v_time_1, v_star_time_0, v_star_time_1


