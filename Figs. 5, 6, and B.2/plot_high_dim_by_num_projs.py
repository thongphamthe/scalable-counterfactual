import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science'])
import dill

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

data = "gamma"

with open("./result_file/" + "high_dim_" + data + ".pkl", 'rb') as file:
    result = dill.load(file)


number_of_repeat = 10

dim_array = [2,10,50,100]

num_iter_array = [1,5,10,25,50,100,200]
#run_vector = ["full","CiC","maxSW_sampling","maxSW_adam"]

run_vector = ["maxSW_sampling","maxSW_adam"]

label_array = ["Proposed (ROT)","$\overline{\\text{ROT}}$"]

choose_iter_array = [1,5,10,25,50,100,200]

invest_d = [2,100]

mean_array = np.zeros((len(invest_d),len(choose_iter_array),len(label_array)))
sd_array   = np.zeros((len(invest_d),len(choose_iter_array),len(label_array)))
mean_array_emd = np.zeros((len(invest_d), len(choose_iter_array),len(label_array)))
sd_array_emd   = np.zeros((len(invest_d), len(choose_iter_array),len(label_array)))

use_m = 9 # dataset number:9,

for d in range(len(invest_d)):
    for jj in range(len(choose_iter_array)):
        for ii in range(len(label_array)):
            values_time = np.zeros((number_of_repeat))
            values_emd = np.zeros((number_of_repeat))

            values_time = result[invest_d[d]][use_m][0][run_vector[ii]][choose_iter_array[jj]]
            values_emd  = result[invest_d[d]][use_m][1][run_vector[ii]][choose_iter_array[jj]]
            mean_array[d][jj][ii]     = np.mean(values_time)
            sd_array[d][jj][ii]       = np.std(values_time)
            mean_array_emd[d][jj][ii] = np.mean(values_emd)
            sd_array_emd[d][jj][ii]   = np.std(values_emd)





pparam_time = dict(xlabel='', ylabel= '')


pparam_emd = dict(xlabel= '', ylabel= '')

color_vec = ["r","black"]
style_line_vec = ["dashed","dashed"]
style_ci_vec = ["solid","solid"]
marker_array = ["o","x"]

#, 'ieee'
width = 1.15
fig = plt.figure(constrained_layout=True,figsize = (9,2))
subfig = fig.subfigures(nrows = 1, ncols = 3,width_ratios=[4,4,1],wspace = 0.2)


font_size = 14
ax = {}
temp = subfig[0].subplots(1,2)
ax[0] = temp
temp = subfig[1].subplots(1,2)
ax[1] = temp

ax[2] = subfig[2]
line_list = list()

subfig[0].suptitle("$d = 2$",fontsize = font_size)
subfig[1].suptitle("$d = 100$", fontsize = font_size)

with plt.style.context(['science', 'ieee']):
    for d in range(len(invest_d)):
        for p in range(len(label_array)):
            temp_fig = ax[d][0].plot(num_iter_array, mean_array[d,:,p], label = label_array[p],
                    marker = marker_array[p], markersize = 4, color = color_vec[p])[0]
            if d > 0:
                line_list.append(temp_fig)
            upper = mean_array[d,:,p] + 2 * sd_array[d,:,p]
            lower = mean_array[d,:,p] - 2 * sd_array[d,:,p]
            for i in range(len(num_iter_array)):
                ax[d][0].plot([num_iter_array[i], num_iter_array[i]], [upper[i], lower[i]],color=color_vec[p], linestyle = "solid")
                ax[d][0].plot([num_iter_array[i] * width, num_iter_array[i] / width], [upper[i], upper[i]],color=color_vec[p], linestyle = "solid")
                ax[d][0].plot([num_iter_array[i] * width, num_iter_array[i] / width], [lower[i], lower[i]],color=color_vec[p], linestyle = "solid")
    



        ax[d][0].set_xticks(num_iter_array)
   
        ax[d][0].set(**pparam_time)
        
        ax[d][0].set_xscale('log')
        xticklabels = ax[d][0].get_xticklabels()
        yticklabels = ax[d][0].get_yticklabels()

        ax[d][0].set_xlabel("$k$",fontsize = font_size)
        ax[d][0].set_ylabel("Running time (in ms)", fontsize=font_size)
        ax[d][0].set_xticklabels(xticklabels, fontsize=font_size)
        ax[d][0].set_yticklabels(yticklabels, fontsize=font_size)

    for d in range(len(invest_d)):
        for p in range(len(label_array)):
            temp_fig = ax[d][1].plot(num_iter_array, mean_array_emd[d, :, p], label=label_array[p],
                                     marker=marker_array[p], markersize=4, color=color_vec[p])[0]
           
            upper = mean_array_emd[d, :, p] + 2 * sd_array_emd[d, :, p]
            lower = mean_array_emd[d, :, p] - 2 * sd_array_emd[d, :, p]
            for i in range(len(num_iter_array)):
                ax[d][1].plot([num_iter_array[i], num_iter_array[i]], [upper[i], lower[i]], color=color_vec[p],
                              linestyle="solid")
                ax[d][1].plot([num_iter_array[i] * width, num_iter_array[i] / width], [upper[i], upper[i]],
                              color=color_vec[p], linestyle="solid")
                ax[d][1].plot([num_iter_array[i] * width, num_iter_array[i] / width], [lower[i], lower[i]],
                              color=color_vec[p], linestyle="solid")
        

        ax[d][1].legend = None
        ax[d][1].set_xticks(num_iter_array)
       
        ax[d][1].set(**pparam_time)
        
        ax[d][1].set_xscale('log')
        xticklabels = ax[d][0].get_xticklabels()
        yticklabels = ax[d][0].get_yticklabels()

        ax[d][1].set_xlabel("$k$",fontsize = font_size)
        ax[d][1].set_ylabel('$OT(\cdot,\widetilde{\mu_{1}^{\\texttt{T}^*}})$', fontsize=font_size)
        ax[d][1].set_xticklabels(xticklabels, fontsize=font_size)
        ax[d][1].set_yticklabels(yticklabels, fontsize=font_size)


ax[2].legend(handles = line_list,
            labels=label_array,  
              bbox_to_anchor=(-0.8,0.75),
               fontsize=font_size,
               loc="center left", ncol=1)

fig.show()

fig.savefig("./images/manuscript/varying_number_of_projections.png",dpi = 600)
