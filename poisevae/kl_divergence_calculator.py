import torch

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class KLDN01:
    __version__ = 3.1 # lambda -> nu
    
    def __init__(self, latent_dims, enc_config, reduction, device=_device):
        self.latent_dims = latent_dims
        self.enc_config = enc_config
        self.reduction = reduction
        self.device = device
        
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
        T = torch.cat((z, torch.square(z)), -1)
        Tp = torch.matmul(T, G)
        mid = G.shape[1] // 2
        
        var = self.var_calc(Tp[..., mid:], nu2)
        mean = self.mean_calc(Tp[..., :mid], var, nu1)
        return torch.square(mean) + var - torch.log(var) - 1
        
    def calc(self, G, z, z_priors, nu1=None, nu2=None, mu=None, var=None, 
             nu1_=None, nu2_=None, mu_=None, var_=None):
        nu1, nu2 = init_posterior(nu1, nu2, mu, var, self.enc_config)
        # nu1_, nu2_ = init_posterior(nu1_, nu2_, mu_, var_, self.enc_config)
        # nu1, nu2 = new_nu(nu1, nu1_), new_nu(nu2, nu2_)
        part0 = self.value_calc(z[1], G.t(), nu1[0].unsqueeze(1), nu2[0].unsqueeze(1)).sum(dim=1)
        part1 = self.value_calc(z[0], G, nu1[1].unsqueeze(1), nu2[1].unsqueeze(1)).sum(dim=1)
        res = (part0 + part1) * 0.5
        if self.reduction == 'sum':
            return res.sum()
        elif self.reduction == 'mean':
            return res.mean()
        else:
            return res
    
    
class KLDDerivative:
    __version__ = 2.3 # lambda -> nu
    
    def __init__(self, latent_dims, enc_config, reduction, device=_device):
        self.latent_dims = latent_dims
        self.enc_config = enc_config
        self.reduction = reduction
        self.device = device
        
    def calc(self, G, z, z_priors, nu1=None, nu2=None, mu=None, var=None,
             nu1_=None, nu2_=None, mu_=None, var_=None):
        nu1, nu2 = init_posterior(nu1, nu2, mu, var, self.enc_config)
        # nu1_, nu2_ = init_posterior(nu1_, nu2_, mu_, var_, self.enc_config)
        # nu1, nu2 = new_nu(nu1, nu1_), new_nu(nu2, nu2_)
        ## Creating Sufficient statistics
        T_prior = torch.cat((z_priors[0], torch.square(z_priors[0])), -1)
        Tp_prior = torch.cat((z_priors[1], torch.square(z_priors[1])), -1)
        T_post = torch.cat((z[0], torch.square(z[0])), -1)
        Tp_post = torch.cat((z[1], torch.square(z[1])), -1)

        nu = torch.cat((nu1[0], nu2[0]), -1) if nu1[0] is not None else \
             torch.zeros(T_post.shape[0], T_post.shape[2]).to(self.device)
        nup = torch.cat((nu1[1], nu2[1]), -1) if nu1[1] is not None else \
              torch.zeros(Tp_post.shape[0], Tp_post.shape[2]).to(self.device)
        # print('zpost max:', torch.abs(z[0]).max().item(), 'zpost mean:', torch.abs(z[0]).mean().item())
        # print('zpostp max:', torch.abs(z[1]).max().item(), 'zpostp mean:', torch.abs(z[1]).mean().item())
        # print('zprior max:', torch.abs(z_priors[0]).max().item(), 'zprior mean:', torch.abs(z_priors[0]).mean().item())
        # print('zpriorp max:', torch.abs(z_priors[1]).max().item(), 'zpriorp mean:', torch.abs(z_priors[1]).mean().item())
        
        with torch.no_grad():
            T_post_diff = T_post - T_post.mean(1, keepdim=True)
            Tp_post_diff = Tp_post - Tp_post.mean(1, keepdim=True)
            n = T_post.shape[1]
            assert Tp_post.shape[1] == n
            n = n - 1 if n > 1 else 1
            cov = torch.bmm(T_post_diff.transpose(1, 2), T_post_diff) / n
            covp = torch.bmm(Tp_post_diff.transpose(1, 2), Tp_post_diff) / n
            # dnu = torch.bmm(nu.unsqueeze(1), cov).squeeze(1)
            # dnup = torch.bmm(nup.unsqueeze(1), covp).squeeze(1)
            
            T_prior_outer = (T_prior.unsqueeze(-1) * Tp_prior.unsqueeze(-2))
            T_post_outer = (T_post.unsqueeze(-1) * Tp_post.unsqueeze(-2))
            # ET_prior_outer, ET_post_outer = T_prior_outer.mean(1), T_post_outer.mean(1)
            nuT = (nu.unsqueeze(1) * T_post).sum(-1, keepdim=True).unsqueeze(-1)
            nupTp = (nup.unsqueeze(1) * Tp_post).sum(-1, keepdim=True).unsqueeze(-1) # (batch, n_gibbs, 1,1)
            dG = (T_post_outer * (nuT + nupTp)).mean(1) \
               - T_post_outer.mean(1) * (nuT + nupTp).mean(1) \
               + T_prior_outer.mean(1) - T_post_outer.mean(1)
            
        # part0 = (dnu * nu).sum(dim=-1)
        # part1 = (dnup * nup).sum(dim=-1)
        dnu = torch.bmm(nu.unsqueeze(1), cov).squeeze(1)
        dnup = torch.bmm(nup.unsqueeze(1), covp).squeeze(1)
        part0 = 0.5 * (dnu * nu).sum(dim=-1)
        part1 = 0.5 * (dnup * nup).sum(dim=-1)
        part2 = (dG * G).sum(dim=(-1, -2))#.detach()
        
        # part2 = (T_prior_outer.detach() * G).sum(dim=(-1, -2)) - \
        #         (T_post_outer.detach() * G).sum(dim=(-1, -2))
        
        # T_post_mean, Tp_post_mean = T_post.mean(1), Tp_post.mean(1)
        
#         part0 = (nu * T_post_mean).sum(dim=-1) + \
#                 (nup * Tp_post_mean).sum(dim=-1)
        
#         part1 = (nu * T_post_mean.detach()).sum(dim=-1) + \
#                 (nup * Tp_post_mean.detach()).sum(dim=-1)
        
#         part2 = ((T_prior_outer - T_post_outer) * G.unsqueeze(0)).sum(dim=(-1, -2))
        
        # print('p0 mean', torch.abs(part0.detach()).mean().item(), 'p1 mean', torch.abs(part1.detach()).mean().item())
        # print('p2 mean', torch.abs(part2.detach()).mean().item(), 'dg mean', torch.abs((T_prior_outer - T_post_outer).detach()).mean().item())
        # print('tpo mean', torch.abs(T_prior_outer.detach()).mean().item(), 'tqo mean', torch.abs(T_post_outer.detach()).mean().item())
        assert len(part0.shape) == 1 and len(part1.shape) == 1 and len(part2.shape) == 1
        if self.reduction == 'mean':
            return part0.mean() + part1.mean() + part2.mean()
        elif self.reduction == 'sum':
            return part0.sum() + part1.sum() + part2.sum()
        else:
            raise NotImplementedError
