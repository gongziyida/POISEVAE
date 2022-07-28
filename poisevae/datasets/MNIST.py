import torch
import scipy.io as sio
import random
import numpy as np
import torch.nn.functional as F

class MNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path):
        self.mnist_pt_path = mnist_pt_path            
        # Load the pt for MNIST 
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        
    def __len__(self):
        return len(self.mnist_data)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        mnist_img, mnist_target = self.mnist_data[index].view(-1)/255, int(self.mnist_targets[index])
        mnist_target = F.one_hot(torch.tensor(mnist_target),num_classes=10)
        # mnist_img = torch.cat((mnist_img,mnist_target),0)
        return mnist_img,mnist_target