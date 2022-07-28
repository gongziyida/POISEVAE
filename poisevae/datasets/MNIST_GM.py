import torch
import random
import numpy as np
from torch.distributions import multivariate_normal as mv

class MNIST_GM(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, sample_size=800, var=1/5, radius=3, data_augment=1):

        self.mnist_pt_path = mnist_pt_path
            
        # Load the pt for MNIST
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        self.mnist_data = self.mnist_data[(self.mnist_targets!=0)&(self.mnist_targets!=9),:,:]
        self.mnist_targets = self.mnist_targets[(self.mnist_targets!=0)&(self.mnist_targets!=9)]
        
        # Generate Gaussian mixtures
        self.gm_var = var * torch.eye(2,dtype=torch.float32)
        angles = np.pi * np.arange(8) / 4
        self.gm_locs = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
        self.gms = [mv.MultivariateNormal(torch.tensor(mu, dtype=torch.float32), self.gm_var) for mu in self.gm_locs]
        self.gm_data = torch.cat([d.sample((sample_size, )) for d in self.gms], dim=0)
        self.gm_targets = np.repeat(np.arange(1, 9, dtype=np.int32), sample_size)
        
        self.gauss_target_idx_mapping = self.process_gauss_labels()
        
        self.data_augment = data_augment
        
    def process_gauss_labels(self):
        numbers_dict = { 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[]}
        for i in range(len(self.gm_targets)):
            gauss_target = self.gm_targets[i]
            numbers_dict[gauss_target].append(i)
        return numbers_dict
    
    def __len__(self):
        return len(self.mnist_data) * self.data_augment - 1
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            Modality 1: 1-8 MNIST
            Modality 2: 8 Gaussian distributions
        """
        idx_mnist = int(np.floor(index / self.data_augment))
        mnist_img, mnist_target = self.mnist_data[idx_mnist], int(self.mnist_targets[idx_mnist])
        indices_list = self.gauss_target_idx_mapping[mnist_target]
        
        # Randomly pick an index from the indices list
        idx = random.choice(indices_list)
        gauss_data = self.gm_data[idx]
        
        gauss_target = self.gm_targets[idx]
        return mnist_img.view(-1)/255, gauss_data, mnist_target, gauss_target