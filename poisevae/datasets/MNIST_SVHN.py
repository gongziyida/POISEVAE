import torch
import scipy.io as sio
import random
import numpy as np

class MNIST_SVHN(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, svhn_mat_path, sampler_mnist=None, sampler_svhn=None, reverse=False):
        self.mnist_pt_path, self.svhn_mat_path = mnist_pt_path, svhn_mat_path
        self.sampler_mnist, self.sampler_svhn = sampler_mnist, sampler_svhn
            
        # Load the pt for MNIST and mat for SVHN 
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        
        # Reading the SVHN data
        svhn_mat_info = sio.loadmat(self.svhn_mat_path)
        self.svhn_data = svhn_mat_info['X']
        # the svhn dataset assigns the class label "10" to the digit 0
        self.svhn_targets = svhn_mat_info['y'].astype(np.int64).squeeze() % 10
        self.svhn_data = np.transpose(self.svhn_data, (3, 2, 0, 1))
        
        self.reverse = reverse
        
        if sampler_mnist is None:
            # Now we have the svhn data and the SVHN Labels, for each index get the classes
            self.svhn_target_idx_mapping = self._process_svhn_labels()
            self.__len__ = lambda: len(self.mnist_data)
        else:
            self.__len__ = lambda: len(self.sampler_mnist)
        
    def _process_svhn_labels(self):
        numbers_dict = {0: [], 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[], 9:[]}
        for i in range(len(self.svhn_targets)):
            svhn_target = self.svhn_targets[i]
            numbers_dict[svhn_target].append(i)
        return numbers_dict
        
    def __len__(self):
        if self.sampler_mnist is None:
            return len(self.mnist_data) - 1
        else:
            return len(self.sampler_mnist) - 1
        
    def __getitem__(self, index):
        if self.sampler_mnist is None:
            mnist_img, mnist_target = self.mnist_data[index], int(self.mnist_targets[index])
            indices_list = self.svhn_target_idx_mapping[mnist_target]

            # Randomly pick an index from the indices list
            idx = random.choice(indices_list)
            svhn_img = self.svhn_data[idx]
            svhn_target = self.svhn_targets[idx]
        else:
            mnist_img = self.mnist_data[self.sampler_mnist[index]]
            mnist_target = int(self.mnist_targets[self.sampler_mnist[index]])

            svhn_img = self.svhn_data[self.sampler_svhn[index]]
            svhn_target = int(self.svhn_targets[self.sampler_svhn[index]])

        if self.reverse:
            return svhn_img/255, mnist_img.view(-1)/255, mnist_target, svhn_target
        else:
            return mnist_img.view(-1)/255, svhn_img/255, mnist_target, svhn_target
    
#         rand_num = random.random()
#         if rand_num < 0.2: # Send both
#             return mnist_img.view(-1)/255, svhn_img/255, mnist_target, svhn_target
#         elif rand_num > 0.6: # Send SVHN only
#             return torch.zeros_like(mnist_img), svhn_img/255, mnist_target, svhn_target
#         else: # Send MNIST only
#             return mnist_img.view(-1)/255, torch.zeros_like(svhn_img), mnist_target, svhn_target
        