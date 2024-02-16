import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science','no-latex'])
import dill


# first dimension is n_array, second dimension is replications (10)
# then the object is an array of dim 26 x 2
# column 0 is running time
# column 1 is error
# "full","CiC","maxSW_sampling","maxSW_adam","SW","sinkhorn"
# full: 0
# CiC: 1
# maxsw_sampling(proposed):2 (1), 3 (2), 4 (10), 5(50), 6 (100), 7 (200), 8 (500)
# maxsw_adam: 9-15
# SW: 16-22
# Sinkhorn: 23-25
# for SW-based, there are 7 values: [1,5,10,25,50,100,200]
# for sinkhorn: there are 3 values: [10,30,90]

n_array        = [5000,10000,15000,20000]
num_iter_array = [1,2,10,50,100,200,500]
m       = 10
#run_vector = ["full","CiC","maxSW_sampling","maxSW_adam","SW"]


# num_iter = 10

# for the main text, plot only sinkhorn (30)

choose = [0,1,4,24]

label_array = ["OT", "CiC", "Proposed", "Sinkhorn"]
marker_array = ["o","p","s",">"]


font_size = 14
cap_size = 8
xlabel_size = 10
plt.rcParams['text.usetex'] = True
ax = {}
fig = plt.figure(layout='constrained',figsize = (9,2.5))
subfig = fig.subfigures(nrows = 2, ncols = 1,height_ratios=[8,1],hspace = 0.05)
temp_first = subfig[0].subfigures(nrows = 1, ncols= 2,wspace = 0.1, hspace = 0.05)
temp = temp_first[0].subplots(1,2)
ax[0] = temp
temp = temp_first[1].subplots(1,2)
ax[1] = temp
ax[2] = subfig[1]
temp_first[0].suptitle("Gamma",fontsize = font_size)
temp_first[1].suptitle("Gaussian mixture", fontsize =font_size)

pparam_time = dict(xlabel='', ylabel= '')


pparam_emd = dict(xlabel='', ylabel= '')

x_time_ticks = [5000,10000,15000,20000]

width = 300
color_vec = ["r","black","blue","gray","gray","gray"]

data = "mixtureG"

with open("./result_file/n_" + data + "_result.pkl", 'rb') as file:
    result = dill.load(file)
mean_array = np.zeros((len(n_array),len(choose)))
sd_array   = np.zeros((len(n_array),len(choose)))

mean_array_emd = np.zeros((len(n_array),len(choose)))
sd_array_emd   = np.zeros((len(n_array),len(choose)))

for n in range(len(n_array)):
    for i in range(len(choose)):
        temp_time = []
        temp_emd = []
        for j in range(m):
            temp_time.append(result[n][j][choose[i]][0]) # time
            temp_emd.append(result[n][j][choose[i]][1])  # emd

        mean_array[n,i]  = np.mean(temp_time)
        sd_array[n,i]    = np.std(temp_time)
        mean_array_emd[n, i] = np.mean(temp_emd)
        sd_array_emd[n, i]   = np.std(temp_emd)

line_list = list()
with plt.style.context(['science', 'ieee']):

    for p in range(len(choose)):
        temp_fig = ax[0][0].plot(n_array, mean_array[:,p], label = label_array[p],
                marker = marker_array[p], markersize = 4, color = color_vec[p])[0]
        line_list.append(temp_fig)
        upper = mean_array[:,p] + 2 * sd_array[:,p]
        lower = mean_array[:,p] - 2 * sd_array[:,p]
        for i in range(len(n_array)):
            ax[0][0].plot([n_array[i], n_array[i]], [upper[i], lower[i]],color=color_vec[p], linestyle = "solid")
            ax[0][0].plot([n_array[i] - width, n_array[i] + width], [upper[i], upper[i]],color=color_vec[p], linestyle = "solid")
            ax[0][0].plot([n_array[i] - width, n_array[i] + width], [lower[i], lower[i]],color=color_vec[p], linestyle = "solid")
   
    ax[0][0].legend = None
    ax[0][0].set_xticks(x_time_ticks)
   
    ax[0][0].set(**pparam_time)
    ax[0][0].set_yscale('log')
    xticklabels = ax[0][0].get_xticklabels()
    yticklabels = ax[0][0].get_yticklabels()
    ax[0][0].set_xlabel("$n$",fontsize = font_size)
    ax[0][0].set_ylabel("Running time (in ms)", fontsize=font_size)
    ax[0][0].set_xticklabels(xticklabels, fontsize=xlabel_size)
    ax[0][0].set_yticklabels(yticklabels, fontsize=font_size)



with plt.style.context(['science', 'ieee']):
    for p in range(len(choose)):
        ax[0][1].plot(n_array, mean_array_emd[:, p], label=label_array[p], marker=marker_array[p],
                markersize=4,color = color_vec[p])
        upper = mean_array_emd[:, p] + 2 * sd_array_emd[:, p]
        lower = mean_array_emd[:, p] - 2 * sd_array_emd[:, p]
        for i in range(len(n_array)):
            ax[0][1].plot([n_array[i], n_array[i]], [upper[i], lower[i]], color=color_vec[p], linestyle = "solid")
            ax[0][1].plot([n_array[i] - width, n_array[i] + width], [upper[i], upper[i]], color=color_vec[p], linestyle = "solid")
            ax[0][1].plot([n_array[i] - width, n_array[i] + width], [lower[i], lower[i]], color=color_vec[p], linestyle = "solid")
    pos = ax[0][1].get_position()
   
    ax[0][1].set(**pparam_emd)
    ax[0][1].set_yscale('log')
    ax[0][1].legend = None
    ax[0][1].set_xticks(x_time_ticks)
    ax[0][1].set_xlabel("$n$",fontsize = font_size)
    ax[0][1].set_ylabel("$OT(\cdot,\widetilde{\mu_{1}^{\\texttt{T}^*}})$", fontsize=font_size)
    xticklabels = ax[0][1].get_xticklabels()
    yticklabels = ax[0][1].get_yticklabels()
    ax[0][1].set_xticklabels(xticklabels, fontsize=xlabel_size)
    ax[0][1].set_yticklabels(yticklabels, fontsize=font_size)

   


data = "gamma"

with open("./result_file/n_" + data + "_result.pkl", 'rb') as file:
    result = dill.load(file)

mean_array = np.zeros((len(n_array),len(choose)))
sd_array   = np.zeros((len(n_array),len(choose)))

mean_array_emd = np.zeros((len(n_array),len(choose)))
sd_array_emd   = np.zeros((len(n_array),len(choose)))

for n in range(len(n_array)):
    for i in range(len(choose)):
        temp_time = []
        temp_emd = []
        for j in range(m):
            temp_time.append(result[n][j][choose[i]][0]) # time
            temp_emd.append(result[n][j][choose[i]][1])  # emd

       
        mean_array[n,i]  = np.mean(temp_time)
        sd_array[n,i]    = np.std(temp_time)
        mean_array_emd[n, i] = np.mean(temp_emd)
        sd_array_emd[n, i]   = np.std(temp_emd)


pparam_time = dict(xlabel='', ylabel= '')


pparam_emd = dict(xlabel='', ylabel= '')

x_time_ticks = [5000,10000,15000,20000]

width = 300
color_vec = ["r","black","blue","gray","gray","gray"]

with plt.style.context(['science', 'ieee']):
    for p in range(len(choose)):
        temp_fig = ax[1][0].plot(n_array, mean_array[:,p], label = label_array[p],
                marker = marker_array[p], markersize = 4, color = color_vec[p])[0]
        upper = mean_array[:,p] + 2 * sd_array[:,p]
        lower = mean_array[:,p] - 2 * sd_array[:,p]
        for i in range(len(n_array)):
            ax[1][0].plot([n_array[i], n_array[i]], [upper[i], lower[i]],color=color_vec[p], linestyle = "solid")
            ax[1][0].plot([n_array[i] - width, n_array[i] + width], [upper[i], upper[i]],color=color_vec[p], linestyle = "solid")
            ax[1][0].plot([n_array[i] - width, n_array[i] + width], [lower[i], lower[i]],color=color_vec[p], linestyle = "solid")
    
    ax[1][0].legend = None
    ax[1][0].set_xticks(x_time_ticks)
    
    ax[1][0].set(**pparam_time)
    ax[1][0].set_yscale('log')
    xticklabels = ax[1][0].get_xticklabels()
    yticklabels = ax[1][0].get_yticklabels()
    ax[1][0].set_xlabel("$n$",fontsize = font_size)
    ax[1][0].set_ylabel("Running time (in ms)", fontsize=font_size)
    ax[1][0].set_xticklabels(xticklabels, fontsize=xlabel_size)
    ax[1][0].set_yticklabels(yticklabels, fontsize=font_size)
    

with plt.style.context(['science', 'ieee']):
    
    for p in range(len(choose)):
        ax[1][1].plot(n_array, mean_array_emd[:, p], label=label_array[p], marker=marker_array[p],
                markersize=4,color = color_vec[p])
        upper = mean_array_emd[:, p] + 2 * sd_array_emd[:, p]
        lower = mean_array_emd[:, p] - 2 * sd_array_emd[:, p]
        for i in range(len(n_array)):
            ax[1][1].plot([n_array[i], n_array[i]], [upper[i], lower[i]], color=color_vec[p], linestyle = "solid")
            ax[1][1].plot([n_array[i] - width, n_array[i] + width], [upper[i], upper[i]], color=color_vec[p], linestyle = "solid")
            ax[1][1].plot([n_array[i] - width, n_array[i] + width], [lower[i], lower[i]], color=color_vec[p], linestyle = "solid")
    pos = ax[1][1].get_position()
  
    ax[1][1].set(**pparam_emd)
    ax[1][1].set_yscale('log')
    ax[1][1].legend = None
    ax[1][1].set_xticks(x_time_ticks)
    ax[1][1].set_xlabel("$n$",fontsize = font_size)
    ax[1][1].set_ylabel("$OT(\cdot,\widetilde{\mu_{1}^{\\texttt{T}^*}})$", fontsize=font_size)
    xticklabels = ax[1][1].get_xticklabels()
    yticklabels = ax[1][1].get_yticklabels()
    ax[1][1].set_xticklabels(xticklabels, fontsize=xlabel_size)
    ax[1][1].set_yticklabels(yticklabels, fontsize=font_size)


ax[2].legend(handles = line_list, labels = label_array,
             fontsize = 14,loc = "center",ncol = 4)

fig.show()
fig.savefig("./images/manuscript/varying_n_combine_plot.png",dpi = 600)


