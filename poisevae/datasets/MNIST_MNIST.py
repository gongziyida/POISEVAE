import torch
import scipy.io as sio
import random
import numpy as np
import torch.nn.functional as F

class MNIST_MNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, special_idx=None, shuffle=True):
        self.mnist_pt_path = mnist_pt_path            
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        
        self.shuffle = shuffle
        if shuffle:
            self.mnist_target_idx_mapping = self.process_mnist_labels()
            
        if special_idx is not None:
            if not hasattr(special_idx, '__iter__'):
                special_idx = [special_idx]
            idx = torch.isin(self.mnist_targets, torch.tensor(special_idx))
            self.mnist_data = self.mnist_data[idx]
            self.mnist_targets = self.mnist_targets[idx]
        
    def process_mnist_labels(self):
        numbers_dict = {0: [], 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[], 9:[]}
        for i in range(len(self.mnist_targets)):
            mnist_target = self.mnist_targets[i].item()
            numbers_dict[mnist_target].append(i)
        return numbers_dict
        
    def __len__(self):
        return len(self.mnist_data)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        if self.shuffle:
            mnist_img_1, mnist_target_1 = self.mnist_data[index], int(self.mnist_targets[index])
            indices_list = self.mnist_target_idx_mapping[(mnist_target_1)]
            # Randomly pick an index from the indices list
            idx = random.choice(indices_list)

            mnist_img_2 = self.mnist_data[idx]
            mnist_target_2 = int(self.mnist_targets[idx])
            return mnist_img_1/255, mnist_img_2/255, mnist_target_1, mnist_target_2
        else:
            mnist_img, mnist_target = self.mnist_data[index]/255, int(self.mnist_targets[index])
            return mnist_img, mnist_img, mnist_target, mnist_target