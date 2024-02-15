import torch.nn as nn
import torch
import ot

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


class Slicer(nn.Module):
    def __init__(self,d,device='cuda',act='linear'):
        super(Slicer, self).__init__()
        self.d=d
        self.device=device
        self.act= act
        self.U_list = nn.Sequential()
        self.U_list.add_module("Linear",nn.Linear(self.d,1,bias=False))
        self.reset() # reset random weights

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        out = self.U_list(x)
        return out
    def project_parameters(self):
        for U in self.U_list.modules():
            if isinstance(U, nn.Linear):
                U.weight.data = U.weight/torch.sqrt(torch.sum(U.weight**2,dim=1,keepdim=True))

    def reset(self):
        for U in self.U_list.modules():
            if isinstance(U, nn.Linear):
                U.weight.data = torch.randn(U.weight.shape)
        self.project_parameters()

def MaxSW(X, Y, slicer,slicer_optimizer,num_iter):
    import numpy as np
    import time
    slicer.train()
    slicer.reset()
    Xdetach=X.detach()
    Ydetach=Y.detach()
    total_time = 0
    #print(num_iter)
    for i in range(num_iter):
        #print(i)
        start = time.perf_counter_ns()
        outX= slicer(Xdetach)
        outY= slicer(Ydetach)
        #print(outX.shape)
        #print(outY.shape)
        #print(outX)
        #print(outY)

        proj_source_sorted = torch.argsort(outX,dim = 0)
        proj_target_sorted = torch.argsort(outY, dim = 0)
        #print(proj_source_sorted)
        #print(outY[proj_target_sorted])
        #print("reach here!")
        negativehsw = -1*torch.sum(torch.abs(outY[proj_target_sorted] - outX[proj_source_sorted]))
        #print(negativehsw)
        slicer_optimizer.zero_grad()
        negativehsw.backward()
        slicer_optimizer.step()
        slicer.project_parameters()
        end = time.perf_counter_ns()
        total_time += (end - start)
    slicer.eval()
    outX = slicer(X)
    outY = slicer(Y)

    proj_source_sorted = torch.argsort(outX,dim = 0)
    #print(proj_source_sorted)

    proj_target_sorted = torch.argsort(outY,dim = 0)
    #print(proj_target_sorted)
    for U in slicer.U_list.modules():
        if isinstance(U, nn.Linear):
            project_vec = U.weight
    return  project_vec,proj_source_sorted,proj_target_sorted,torch.sum(abs(outY[proj_target_sorted] - outX[proj_source_sorted])),total_time

def one_dimensional_Wasserstein_prod(X_prod,Y_prod,p):
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=0), 1.0 / p)
    wasserstein_distance = torch.pow(wasserstein_distance, p).mean()
    return wasserstein_distance

def one_dimensional_Wasserstein(X,Y,theta):
    X_prod = torch.matmul(X, theta)
    Y_prod = torch.matmul(Y, theta)
    #print(X_prod)
    proj_source_sorted = torch.argsort(X_prod, dim=0)
    #print(proj_source_sorted)
    #print(X_prod[proj_source_sorted])
    proj_target_sorted = torch.argsort(Y_prod, dim=0)

    wasserstein_distance= torch.sum(torch.abs(Y_prod[proj_target_sorted] - X_prod[proj_source_sorted]))
    return wasserstein_distance