from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel


class GP(ExactGP):
    
    def __init__(self, info, dataset2per, dataset2sem, model2per, hard_per_matrix, hard_sem_matrix, soft_list, train_X, train_Y, likelihood):
        
        super().__init__(train_X, train_Y, likelihood)
        
        self.num_all_datasets = info.num_all_datasets
        self.num_all_models = info.num_all_models
        
        self.mean_module = nn.ModuleList([ConstantMean() for _ in range(self.num_all_datasets * self.num_all_models)])
        self.covar_module_dataset = Dataset_Kernel(dataset2per, dataset2sem)
        self.covar_module_model = Model_Kernel(model2per)
        self.covar_module_instruction = Instruction_Kernel(hard_per_matrix, hard_sem_matrix, soft_list)
        
    def forward(self, X):
        
        groupby_task = defaultdict(list)
        for idx, x in enumerate(X):
            groupby_task[int(x[0].item())].append(idx)
        mean_X = torch.zeros(X.shape[0]).to(X.device)
        for taskID in groupby_task:
            indices = torch.Tensor(groupby_task[taskID]).long()
            mean_X[indices] = self.mean_module[taskID](X[indices,0])
        
        covar_X_dataset = self.covar_module_dataset(X[:,0] // self.num_all_models)
        covar_X_model = self.covar_module_model(X[:,0] % self.num_all_models)
        covar_X_instruction = self.covar_module_instruction(X[:,1:])
        covar_X = covar_X_dataset * covar_X_model * covar_X_instruction
        
        return MultivariateNormal(mean_X, covar_X)
    
    
class Dataset_Kernel(Kernel):
    
    def __init__(self, dataset2per, dataset2sem):
        
        super().__init__()
        
        self.dataset2per = dataset2per
        self.dataset2sem = dataset2sem
        
        self.register_parameter(name="raw_outputscale", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", Positive())
        
    @property
    def outputscale(self):
        
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)    

    @outputscale.setter
    def outputscale(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
        
    def forward(self, X, _, **params):
        
        rep_per = self.dataset2per[X.squeeze().long()]
        cov_per = torch.from_numpy(cosine_similarity(rep_per.cpu())).to(rep_per.device)
        rep_sem = self.dataset2sem[X.squeeze().long()]
        cov_sem = torch.from_numpy(cosine_similarity(rep_sem.cpu())).to(rep_sem.device)
        
        return self.outputscale * cov_per * cov_sem
        
    
class Model_Kernel(Kernel):
    
    def __init__(self, model2per):
        
        super().__init__()
        
        self.model2per = model2per
        
        self.register_parameter(name="raw_outputscale", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", Positive())
        
    @property
    def outputscale(self):
        
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)    

    @outputscale.setter
    def outputscale(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
        
    def forward(self, X, _, **params):
        
        rep_per = self.model2per[X.squeeze().long()]
        cov_per = torch.from_numpy(cosine_similarity(rep_per.cpu())).to(rep_per.device)
        
        return self.outputscale * cov_per

        
class Instruction_Kernel(Kernel):
    
    def __init__(self, hard_per_matrix, hard_sem_matrix, soft_list):
        
        super().__init__()
        
        self.hard_per_matrix = hard_per_matrix
        self.hard_sem_matrix = hard_sem_matrix
        self.soft_list = soft_list
        
        self.inv_add = torch.eye(soft_list.shape[0]).to(soft_list.device)
        self.soft_kernel = ScaleKernel(RBFKernel(ard_num_dims=soft_list.shape[-1]))
    
        self.register_parameter(name="raw_outputscale", parameter=nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_outputscale", Positive())
        
    @property
    def outputscale(self):
        
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)    

    @outputscale.setter
    def outputscale(self, value):
        
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))
        
    def forward(self, X, _, **params):
        
        cov_hard = (self.outputscale * self.hard_sem_matrix * self.hard_per_matrix).double()
        cov_soft_inv = torch.inverse(self.soft_kernel(self.soft_list).evaluate() + 0.0001 * self.inv_add).double()
        cov_X = self.soft_kernel(X, self.soft_list).double()
        
        return (cov_X @ cov_soft_inv @ cov_hard @ cov_soft_inv.T @ cov_X.T).float()