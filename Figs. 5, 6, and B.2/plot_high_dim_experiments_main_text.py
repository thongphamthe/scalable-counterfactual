import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science','no-latex'])
import dill
plt.rcParams['text.usetex'] = True
data = "gamma"

with open("./result_file/" + "high_dim_" + data + ".pkl", 'rb') as file:
    result = dill.load(file)
# result is a dictionary with keys: 2, 10, 50, 100: the dimension d
# in each key, is a list of length 10, i.e., the repetition
# result[2][i] will be a list of length 2, first is the time, second is the emd
# result[2][i][0] or result[2][i][1] will be a dictionary,
# with keys full', 'CiC', 'maxSW_sampling', 'maxSW_adam', 'SW', 'sinkhorn'
# for full and CiC, it will just be a number
# for maxSW_sampling, it will be a dictionary with keys: 1, 5, 10, 25, 50, 100, 200
# for sinkhorn, it will be a dictionary with keys: 10, 30, 90


number_of_repeat = 10
dim_array = [2,10,50,100]
#dim_array = [100]

run_vector = ["full","CiC","maxSW_sampling","sinkhorn"]

label_array = ["OT", "CiC", "Proposed", "Sinkhorn"]
marker_array = ["o","p","s",">"]

choose_iter_array = [10] # key of a dictionary, only for maxSW_sampling

color_vec = ["r","black","blue","gray","gray","gray"]

mean_array = np.zeros((len(dim_array),len(label_array)))
sd_array   = np.zeros((len(dim_array),len(label_array)))

mean_array_emd = np.zeros((len(dim_array),len(label_array)))
sd_array_emd   = np.zeros((len(dim_array),len(label_array)))

for d in range(len(dim_array)):
    for i in range(len(run_vector)):
        if run_vector[i] in ["full","CiC"]:
            values_time = np.zeros((number_of_repeat))
            values_emd = np.zeros((number_of_repeat))
            for use_m in range(number_of_repeat):
                values_time[use_m] = result[dim_array[d]][use_m][0][run_vector[i]][0]
                values_emd[use_m] = result[dim_array[d]][use_m][1][run_vector[i]][0]
            mean_array[d][i] = np.mean(values_time)
            sd_array[d][i] = np.std(values_time)
            mean_array_emd[d][i] = np.mean(values_emd)
            sd_array_emd[d][i] = np.std(values_emd)
        elif run_vector[i] == "maxSW_sampling":
            values_time = np.zeros((number_of_repeat))
            values_emd = np.zeros((number_of_repeat))
            for use_m in range(number_of_repeat):
                values_time[use_m] = result[dim_array[d]][use_m][0][run_vector[i]][10][0] # choose the number of projections as 10
                values_emd[use_m] = result[dim_array[d]][use_m][1][run_vector[i]][10][0] # choose the number of projections as 10
            mean_array[d][i] = np.mean(values_time)
            sd_array[d][i] = np.std(values_time)
            mean_array_emd[d][i] = np.mean(values_emd)
            sd_array_emd[d][i] = np.std(values_emd)
        else: # case of sinkhorn

            # 10:
            #values_time = np.zeros((number_of_repeat))
            #values_emd = np.zeros((number_of_repeat))
            #for use_m in range(number_of_repeat):
            #    values_time[use_m] = result[dim_array[d]][use_m][0][run_vector[i]][10][0]  # choose the number of projections as 10
            #    values_emd[use_m] = result[dim_array[d]][use_m][1][run_vector[i]][10][0]  # choose the number of projections as 10
            #mean_array[d][i] = np.mean(values_time)
            #sd_array[d][i] = np.std(values_time)
            #mean_array_emd[d][i] = np.mean(values_emd)
            #sd_array_emd[d][i] = np.std(values_emd)

            # 30:
            values_time = np.zeros((number_of_repeat))
            values_emd = np.zeros((number_of_repeat))
            for use_m in range(number_of_repeat):
                values_time[use_m] = result[dim_array[d]][use_m][0][run_vector[i]][30][0]  # choose the number of projections as 10
                values_emd[use_m] = result[dim_array[d]][use_m][1][run_vector[i]][30][0]  # choose the number of projections as 10
            mean_array[d][i] = np.mean(values_time)
            sd_array[d][i] = np.std(values_time)
            mean_array_emd[d][i] = np.mean(values_emd)
            sd_array_emd[d][i] = np.std(values_emd)

            # 90:
            #values_time = np.zeros((number_of_repeat))
            #values_emd = np.zeros((number_of_repeat))
            #for use_m in range(number_of_repeat):
            #    values_time[use_m] = result[dim_array[d]][use_m][0][run_vector[i]][90][0]  # choose the number of projections as 10
            #    values_emd[use_m] = result[dim_array[d]][use_m][1][run_vector[i]][90][0]  # choose the number of projections as 10
            #mean_array[d][i + 2] = np.mean(values_time)
            #sd_array[d][i + 2] = np.std(values_time)
            #mean_array_emd[d][i + 2] = np.mean(values_emd)
            #sd_array_emd[d][i + 2] = np.std(values_emd)





pparam_time = dict(xlabel='', ylabel= '')


pparam_emd = dict(xlabel='', ylabel= '')

x_time_ticks = [2,10,50,100]


font_size = 10

ax = {}
fig = plt.figure(layout='constrained',figsize = (6,3.5))
subfig = fig.subfigures(nrows = 2, ncols = 1,height_ratios=[8,1],hspace = 0)
temp = subfig[0].subplots(1,2)
ax[0] = temp
ax[1] = subfig[1]

width = 3
color_vec = ["r","black","blue","gray"]
line_list = list()
with plt.style.context(['science', 'ieee']):
    for p in range(len(label_array)):
        temp_fig = ax[0][0].plot(dim_array, mean_array[:,p], label = label_array[p],
                marker = marker_array[p], markersize = 4, color = color_vec[p])[0]
        line_list.append(temp_fig)
        upper = mean_array[:,p] + 2 * sd_array[:,p]
        lower = mean_array[:,p] - 2 * sd_array[:,p]
        for i in range(len(dim_array)):
            ax[0][0].plot([dim_array[i], dim_array[i]], [upper[i], lower[i]],color=color_vec[p], linestyle = "solid")
            ax[0][0].plot([dim_array[i] - width, dim_array[i] + width], [upper[i], upper[i]],color=color_vec[p], linestyle = "solid")
            ax[0][0].plot([dim_array[i] - width, dim_array[i] + width], [lower[i], lower[i]],color=color_vec[p], linestyle = "solid")
    #ax.legend(title='Method')
    ax[0][0].legend = None
    ax[0][0].set_xticks(x_time_ticks)
    #pos = ax.get_position()
    #ax.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
    #ax.legend(title='Method', loc='center right',
    #          fontsize='small', bbox_to_anchor=(1.65, 0.5))
    #ax.autoscale(tight=True)
    ax[0][0].set(**pparam_time)
    ax[0][0].set_yscale('log')
    xticklabels = ax[0][0].get_xticklabels()
    yticklabels = ax[0][0].get_yticklabels()
    ax[0][0].set_xlabel("$d$",fontsize = font_size)
    ax[0][0].set_ylabel("Running time (in ms)", fontsize=font_size)
    ax[0][0].set_xticklabels(xticklabels, fontsize=font_size)
    ax[0][0].set_yticklabels(yticklabels, fontsize=font_size)
    #fig.savefig(data + '_time.png',dpi = 200)
    #fig.tight_layout()
    #fig.show()

with plt.style.context(['science', 'ieee']):
    #fig, ax = plt.subplots()
    for p in range(len(label_array)):
        ax[0][1].plot(dim_array, mean_array_emd[:, p], label=label_array[p], marker=marker_array[p],
                markersize=4,color = color_vec[p])
        upper = mean_array_emd[:, p] + 2 * sd_array_emd[:, p]
        lower = mean_array_emd[:, p] - 2 * sd_array_emd[:, p]
        for i in range(len(dim_array)):
            ax[0][1].plot([dim_array[i], dim_array[i]], [upper[i], lower[i]], color=color_vec[p], linestyle = "solid")
            ax[0][1].plot([dim_array[i] - width, dim_array[i] + width], [upper[i], upper[i]], color=color_vec[p], linestyle = "solid")
            ax[0][1].plot([dim_array[i] - width, dim_array[i] + width], [lower[i], lower[i]], color=color_vec[p], linestyle = "solid")
    pos = ax[0][1].get_position()
    #ax.legend(title='', fontsize = 16)
    #ax.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
    #ax.legend(title='Method',loc='center right',
    #          fontsize='small',bbox_to_anchor=(1.65, 0.5))
    # ax.autoscale(tight=True)
    ax[0][1].set(**pparam_emd)
    ax[0][1].set_yscale('log')
    ax[0][1].legend = None
    ax[0][1].set_xticks(x_time_ticks)
    ax[0][1].set_xlabel("$d$",fontsize = font_size)
    ax[0][1].set_ylabel("$OT(\cdot,\widetilde{\mu_{1}^{\\texttt{T}^*}})$", fontsize=font_size)
    xticklabels = ax[0][1].get_xticklabels()
    yticklabels = ax[0][1].get_yticklabels()
    ax[0][1].set_xticklabels(xticklabels, fontsize=font_size)
    ax[0][1].set_yticklabels(yticklabels, fontsize=font_size)

    #fig.savefig(data + '_emd.pdf')
    #fig.savefig("./images/"+ data + '_emd.png',dpi = 600)


fig.subplots_adjust(bottom=0.5)
#plt.subplots_adjust(hspace = 0.5)
ax[1].legend(handles = line_list,
              labels = label_array,# marker = marker_array,
               #color = color_vec,
               #bbox_to_anchor=(0.015,-0.02),
               fontsize = font_size,
               loc = "center",ncol = 6)
#fig.tight_layout()
fig.show()
fig.savefig("./images/manuscript/varying_dimmension_gamma.png",dpi = 600)
#fig.savefig('figures/fig2c.jpg', dpi=300)


