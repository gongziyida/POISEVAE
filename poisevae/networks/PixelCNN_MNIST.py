import torch
import torch.nn as nn

class EncMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(EncMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.dim_MNIST = 28 * 28

        self.enc = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(512, 128),
                                 nn.ReLU(inplace=True))
        self.enc_mu_mnist = nn.Linear(128, latent_dim)
        self.enc_var_mnist = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        mu_mnist = self.enc_mu_mnist(x)
        log_var_mnist = self.enc_var_mnist(x)
        return mu_mnist, log_var_mnist
    

class DecMNIST(nn.Module):
    def __init__(self, pixelcnn, color_level):
        super(DecMNIST, self).__init__()
        self.pixelcnn = pixelcnn
        self.color_level = color_level
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, z, x, generate_mode):
        x = x.reshape(z.shape[0], 1, 28, 28).to(z.device)
        if generate_mode is False:
            sample = self.pixelcnn(x, z)
            x = (x.flatten(-3, -1) * (self.color_level - 1)).floor().long()
            return sample, self.ce_loss(sample.flatten(-3, -1), x)
        else:
            shape = [1,28,28]
            sample = self.pixelcnn.sample(x, shape, z, x.device)
            sample = torch.exp(sample)
            return sample, self.mse_loss(sample, x)