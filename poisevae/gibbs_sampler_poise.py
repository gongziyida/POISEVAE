import torch

def init_posterior(nu1, nu2, mu, var, enc_config):
    if enc_config == 'mu/var':
        if mu is not None and var is not None:
            nu1, nu2 = [None, None], [None, None]
            
            for i in (0, 1):
                if mu[i] is not None and var[i] is not None:
                    nu1[i] = mu[i] / var[i]
                    nu2[i] = -1 / (2 * var[i])
            return nu1, nu2

    elif enc_config == 'nu':
        if nu1 is not None and nu2 is not None:
            return nu1, nu2
        
    elif enc_config == 'mu/nu2':
        if mu is not None and nu2 is not None:
            nu1 = [None, None]
            
            for i in (0, 1):
                if mu[i] is not None and nu2[i] is not None:
                    nu1[i] = -2 * mu[i] * nu2[i]
            return nu1, nu2
    # Prior
    return [None, None], [None, None]

def new_nu(nu, nu_tilde):
    nu_ = [None, None]
    for i in (0, 1):
        if (nu[i] is None) and (nu_tilde[1-i] is not None):
            nu_[i] = nu_tilde[1-i]
        elif (nu[i] is not None) and (nu_tilde[1-i] is None):
            nu_[i] = nu[i]
        elif (nu[i] is not None) and (nu_tilde[1-i] is not None):
            nu_[i] = nu[i] + nu_tilde[1-i]
    return nu_

class GibbsSampler:
    __version__ = 1.0 # generator
    def __init__(self, latent_dims, enc_config, dtype=torch.float32):
        self.latent_dims = latent_dims
        self.enc_config = enc_config
        self.dtype = dtype
    
    def var_calc(self, T2, nu2):
        if nu2 is not None:
            return -torch.reciprocal(2 * (-0.5 + nu2 + T2))
        else:
            return -torch.reciprocal(2 * (-0.5 + T2))
    
    def mean_calc(self, T1, var, nu1):
        if nu1 is not None:
            return var * (T1 + nu1)
        else:
            return var * (T1)
        
    def value_calc(self, z, G, nu1, nu2):
        T = torch.cat((z, torch.square(z)), 1)
        Tp = torch.matmul(T, G)
        mid = G.shape[1] // 2
        
        var = self.var_calc(Tp[:, mid:], nu2)
        mean = self.mean_calc(Tp[:, :mid], var, nu1)
        new_z = mean + torch.sqrt(var) * torch.randn_like(var)
        T = torch.cat((new_z, torch.square(new_z)), 1)
        return new_z, T
        
            
    def init_z(self, nu1, nu2, batch_size, device):
        z = []
        for ld, nu1_i, nu2_i in zip(self.latent_dims, nu1, nu2):
            if (nu1_i is None) and (nu2_i is None): # the Nones come together
                if batch_size is None:
                    raise RuntimeError('batch_size must be specified for prior.')
                var = torch.tensor(1)
                mu = torch.tensor(0)
                rand = torch.randn(batch_size, ld).to(device)
            else:
                # if not (nu2_i < 0).all():
                #     raise ValueError('Invalid variance')
                var = -torch.reciprocal(2 * (-0.5 + nu2_i))
                mu = var * (nu1_i)
                rand = torch.randn_like(mu)
            z.append((mu + rand * torch.sqrt(var)).to(device))
        return z
    
    
    def sample(self, G, nu1=None, nu2=None, mu=None, var=None, 
               nu1_=None, nu2_=None, mu_=None, var_=None, 
               n_iterations=15, n_samples=None, batch_size=None):
        nu1, nu2 = init_posterior(nu1, nu2, mu, var, self.enc_config)
        # nu1_, nu2_ = init_posterior(nu1_, nu2_, mu_, var_, self.enc_config)
        # nu1, nu2 = new_nu(nu1, nu1_), new_nu(nu2, nu2_)

        if n_samples is None:
            n_samples = n_iterations
        
        z1, z2 = self.init_z(nu1, nu2, batch_size=batch_size, device=G.device)
        
        assert len(z1.shape) == 2
        assert len(z2.shape) == 2
        z = [torch.zeros(z1.shape[0], n_samples, z1.shape[1]).to(z1.device), 
             torch.zeros(z2.shape[0], n_samples, z2.shape[1]).to(z2.device)]
        T = [torch.zeros(*z[0].shape[:-1], z1.shape[1] * 2).to(z1.device), 
             torch.zeros(*z[1].shape[:-1], z2.shape[1] * 2).to(z2.device)]

        for i in range(n_iterations):
            z1, T1 = self.value_calc(z2, G.t(), nu1[0], nu2[0])
            z2, T2 = self.value_calc(z1, G, nu1[1], nu2[1])
            if i >= n_iterations - n_samples:
                k = i - (n_iterations - n_samples)
                z[0][:, k] = z1
                z[1][:, k] = z2
                T[0][:, k] = T1
                T[1][:, k] = T2

        return z, T
    
    
    def sample_generator(self, G, nu1=None, nu2=None, mu=None, var=None, 
                         n_iterations=1, batch_size=None):
        with torch.no_grad():
            nu1, nu2 = init_posterior(nu1, nu2, mu, var, self.enc_config)

            z = self.init_z(mu=mu, var=var, batch_size=batch_size)

            # TODO: generalize to M > 2
            for i in range(n_iterations):
                z[0], T1 = self.value_calc(z[1], G.t(), nu1[0], nu2[0])
                z[1], T2 = self.value_calc(z[0], G, nu1[1], nu2[1])
                yield z
