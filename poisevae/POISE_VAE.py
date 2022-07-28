import torch
import torch.nn as nn
from .gibbs_sampler_poise import GibbsSampler
from .gradient import KLGradient, RecGradient
# from .gradient_coupled_prior import KLGradient, RecGradient
from .kl_divergence_calculator import KLDDerivative, KLDN01
from numpy import prod, sqrt, log

def _latent_dims_type_setter(lds):
    ret, ret_flatten = [], []
    for ld in lds:
        if hasattr(ld, '__iter__'): # Iterable
            ld_tuple = tuple([i for i in ld])
            if not all(map(lambda i: isinstance(i, int), ld_tuple)):
                raise ValueError('`latent_dim` must be either iterable of ints or int.')
            ret.append(ld_tuple)
            ret_flatten.append(int(prod(ld_tuple)))
        elif isinstance(ld, int):
            ret.append((ld, ))
            ret_flatten.append(ld)
        else:
            raise ValueError('`latent_dim` must be either iterable of ints or int.')
    return ret, ret_flatten


def _func_type_setter(func, num, fname, concept):
    if callable(func):
        ret = [func] * num
    elif hasattr(func, '__iter__'):
        if len(func) != num:
            raise ValueError('Unmatched number of %s and datasets' % concept)
        ret = func
    else:
        raise TypeError('`%s` must be callable or list of callables.' % fname)
    return ret


class POISEVAE(nn.Module):
    
    def __init__(self, encoders, decoders, latent_dims=None, rec_weights=None, mask_missing=None, 
                 batched=True, batch_size=-1, enc_config='nu', KL_calc='derivative_autograd'):
        """
        Parameters
        ----------
        encoders: list of nn.Module
            Each encoder must have an attribute `latent_dim` specifying the dimension of the
            latent space to which it encodes. An alternative way to avoid adding this attribute
            is to specify the `latent_dims` parameter (see below). 
            Note that each `latent_dim` must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
            For now the model only support Gaussian distributions of the encodings. 
            The encoders must output the mean and log variance of the Gaussian distributions.
            
        decoders: list of nn.Module
            The number and indices of decoders must match those of encoders.
        
        latent_dims: iterable of int, optional; default None
            The dimensions of the latent spaces to which the encoders encode. The indices of the 
            entries must match those of encoders. An alternative way to specify the dimensions is
            to add the attribute `latent_dim` to each encoder (see above).
            Note that each entry must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
        
        rec_weights: iterable of float, optional; default None
            The weights of the reconstruction loss of each modality
            
        mask_missing: callable, optional; default None
            Must be of the form `mask_missing(data)` and return the masked data
            The missing data should be None, while the present data should have the same data structures.
            
        batched: bool, default True
            If the data is in batches
            
        batch_size: int, default -1
            Default: automatically determined
        
        enc_config: str, default 'nu'
            The definition of the encoder output, either 'nu' or 'mu/var'
        
        KL_calc: str, default 'derivative_autograd'
            'derivative_autograd', 'derivative_gradient', 'std_normal'
        """
        super(POISEVAE,self).__init__()

        # Modality check
        if len(encoders) != len(decoders):
            raise ValueError('The number of encoders (%d) must match that of decoders (%d).' \
                             % (len(encoders), len(decoders)))
        if len(encoders) > 2:
            raise NotImplementedError('> 3 latent spaces not yet supported.')
        
        # Type check
        if not all(map(lambda x: isinstance(x, nn.Module), (*encoders, *decoders))):
            raise TypeError('`encoders` and `decoders` must be lists of `nn.Module` class.')

        # Flag check
        if enc_config not in ('nu', 'mu/var', 'mu/nu2'):
            raise ValueError('`enc_config` value unreconized.')
        if KL_calc not in ('derivative_autograd', 'derivative_gradient', 'std_normal'): 
            raise ValueError('`KL_calc` value unreconized.')
            
        # Get the latent dimensions
        if latent_dims is not None:
            if not hasattr(latent_dims, '__iter__'): # Iterable
                raise TypeError('`latent_dims` must be iterable.')
            self.latent_dims = latent_dims
        else:
            self.latent_dims = tuple(map(lambda l: l.latent_dim, encoders))
        self.latent_dims, self.latent_dims_flatten = _latent_dims_type_setter(self.latent_dims)
        self.M = len(self.latent_dims)

        self.mask_missing = mask_missing
        
        self.batched = batched
        self._batch_size = batch_size # init
            
        self.rec_weights = rec_weights
        
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        
        self.enc_config = enc_config
        
        self.gibbs = GibbsSampler(self.latent_dims_flatten, enc_config=enc_config)
        
        self.KL_calc = KL_calc
        if KL_calc == 'derivative_autograd':
            self.kl_div = KLDDerivative(self.latent_dims_flatten, reduction='mean', 
                                        enc_config=enc_config)
        elif KL_calc == 'std_normal':
            self.kl_div = KLDN01(self.latent_dims_flatten, reduction='mean', 
                                 enc_config=enc_config)
        
        
        self.g11 = nn.Parameter(torch.randn(*self.latent_dims_flatten))
        self.g22_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten))
        self.g12_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten))
        self.g21_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten))
        
        self._dummy = nn.Parameter(torch.tensor(0.0))
        
        
    def set_mask_missing(self, mask_missing):
        self.mask_missing = mask_missing
        
    def get_G(self):
        g22 = -torch.exp(self.g22_hat)
        g12 = 2 / sqrt(self.latent_dims_flatten[0]) * \
              torch.exp(self.g22_hat / 2 + log(0.5) / 2) * \
              torch.tanh(self.g12_hat) * 0.99
        g21 = 2 / sqrt(self.latent_dims_flatten[1]) * \
              torch.exp(self.g22_hat / 2 + log(0.5) / 2) * \
              torch.tanh(self.g21_hat) * 0.99
        G = torch.cat((torch.cat((self.g11, g12), 1), torch.cat((g21, g22), 1)), 0)
        return G
    
    def encode(self, x, **kwargs):
        """
        Encode the samples from multiple sources
        Parameter
        ---------
        x: list of torch.Tensor
        Return
        ------
        z: list of torch.Tensor
        """
        param1, param2 = [], []
        
        for i, xi in enumerate(x):
            if xi is None:
                param1.append(None)
                param2.append(None)
            else:
                batch_size = xi.shape[0] if self.batched else 1
                ret = self.encoders[i](xi, **kwargs)
                param1.append(ret[0].view(batch_size, -1))
                sign = 1 if self.enc_config == 'mu/var' else -1
                param2.append(sign * torch.exp(ret[1].view(batch_size, -1)))
                # param2.append(sign * nn.functional.softplus(ret[1].view(batch_size, -1)))

        return param1, param2
    
    def decode(self, z, x, **kwargs):
        """
        Unsqueeze the samples from each latent space (if necessary), and decode
        Parameter
        ---------
        z: list of torch.Tensor
        Return
        ------
        x_rec: list of torch.Tensor
        neg_loglike: list of torch.Tensor
        """
        x_rec = []
        neg_loglike = []
        batch_size = z[0].shape[0] if self.batched else 1
        
        len_neg_loglike = 1 # Batch dim
        Gibbs_dim = len(z[0].shape) == 2 + self.batched
        if Gibbs_dim:
            n_samples = z[0].shape[self.batched]
            z = [zi.flatten(0, 1) for zi in z]
            x = [xi.repeat_interleave(n_samples, dim=0) for xi in x]
            len_neg_loglike += 1
            
        for i, (decoder, zi, ld) in enumerate(zip(self.decoders, z, self.latent_dims)):
            zi = zi.view(batch_size * n_samples, *ld) # Match the shape to the output
            x_rec_, neg_loglike_ = decoder(zi, x[i], **kwargs)
            if Gibbs_dim: # Gibbs dimension
                x_rec_ = x_rec_.view(batch_size, n_samples, *x_rec_.shape[1:])
                neg_loglike_ = neg_loglike_.view(batch_size, n_samples, *neg_loglike_.shape[1:])
            x_rec.append(x_rec_)
            
            if len(neg_loglike_.shape) > len_neg_loglike: # Not summing batch and Gibbs dims
                dims = list(range(len_neg_loglike, len(neg_loglike_.shape))) 
                neg_loglike_ = neg_loglike_.sum(dims)
                
            if self.rec_weights is not None:
                neg_loglike_ *= self.rec_weights[i]
                
            neg_loglike.append(neg_loglike_)
        return x_rec, neg_loglike
    
    
    def _sampling_autograd(self, G, param1, param2, n_iterations=30):
        batch_size = self._fetch_batch_size(param1)
        self._batch_size = batch_size
        
        z_priors, T_priors = self.gibbs.sample(G, batch_size=batch_size, n_iterations=n_iterations)
        
        if self.enc_config == 'nu':
            z_posteriors, T_posteriors = self.gibbs.sample(G, nu1=param1, nu2=param2, 
                                                           batch_size=batch_size,
                                                           n_iterations=n_iterations)
            kl = self.kl_div.calc(G, z_posteriors, z_priors, nu1=param1, nu2=param2)
            
        elif self.enc_config == 'mu/var':
            z_posteriors, T_posteriors = self.gibbs.sample(G, mu=param1, var=param2, 
                                                           batch_size=batch_size,
                                                           n_iterations=n_iterations)
            kl = self.kl_div.calc(G, z_posteriors, z_priors, mu=param1, var=param2)
        elif self.enc_config == 'mu/nu2':
            z_posteriors, T_posteriors = self.gibbs.sample(G, mu=param1, nu2=param2, 
                                                           batch_size=batch_size,
                                                           n_iterations=n_iterations)
            kl = self.kl_div.calc(G, z_posteriors, z_priors, mu=param1, nu2=param2)
            # if param1[0] is not None and param1[1] is not None:
            #     assert torch.isnan(param1[0]).sum() == 0
            #     assert torch.isnan(-0.5 / param2[0]).sum() == 0
            
        return z_posteriors, kl
    
    def _sampling_gradient(self, G, param1, param2, n_iterations=30):
        batch_size = self._fetch_batch_size(param1)
        self._batch_size = batch_size
        
        with torch.no_grad():
            z_priors, T_priors = self.gibbs.sample(G.detach(), batch_size=batch_size, n_iterations=n_iterations)
        
        if self.enc_config == 'nu':
            with torch.no_grad():
                z_posteriors, T_posteriors = self.gibbs.sample(G, nu1=param1, nu2=param2, 
                                                               batch_size=batch_size,
                                                               n_iterations=n_iterations)
            nu = torch.cat([param1[0], param2[0]], -1) if param1[0] is not None else \
                 torch.zeros(T_posteriors[0].shape[0], T_posteriors[0].shape[2]).to(self._dummy.device)
            nup = torch.cat([param1[1], param2[1]], -1) if param1[1] is not None else \
                  torch.zeros(T_posteriors[1].shape[0], T_posteriors[1].shape[2]).to(self._dummy.device)
            
        elif self.enc_config == 'mu/var':
            with torch.no_grad():
                z_posteriors, T_posteriors = self.gibbs.sample(G, mu=param1, var=param2, 
                                                               batch_size=batch_size,
                                                               n_iterations=n_iterations)
            nu = torch.cat([param1[0] / param2[0], -0.5 / param2[0]], -1) if param1[0] is not None else \
                 torch.zeros(T_posteriors[0].shape[0], T_posteriors[0].shape[2]).to(self._dummy.device)
            nup = torch.cat([param1[1] / param2[1], -0.5 / param2[1]], -1) if param1[1] is not None else \
                  torch.zeros(T_posteriors[1].shape[0], T_posteriors[1].shape[2]).to(self._dummy.device)
            assert torch.isnan(param1[0]).sum() == 0
            assert torch.isnan(-0.5 / param2[0]).sum() == 0
        elif self.enc_config == 'mu/nu2':
            with torch.no_grad():
                z_posteriors, T_posteriors = self.gibbs.sample(G, mu=param1, nu2=param2, 
                                                               batch_size=batch_size,
                                                               n_iterations=n_iterations)
            nu = torch.cat([-2 * param1[0] * param2[0], param2[0]], -1) if param1[0] is not None else \
                 torch.zeros(T_posteriors[0].shape[0], T_posteriors[0].shape[2]).to(self._dummy.device)
            nup = torch.cat([-2 * param1[1] * param2[1], param2[1]], -1) if param1[1] is not None else \
                  torch.zeros(T_posteriors[1].shape[1], T_posteriors[1].shape[2]).to(self._dummy.device)
            
        return z_priors, z_posteriors, T_priors, T_posteriors, [nu, nup]
    
    
    
    def forward(self, x, n_gibbs_iter=15, kl_weight=1, detach_G=False, enc_kwargs={}, dec_kwargs={}):
        """
        Return
        ------
        results: dict
            z: list of torch.Tensor
                Samples from the posterior distributions in the corresponding latent spaces
            x_rec: list of torch.Tensor
                Reconstructed samples
            param1: list of torch.Tensor
                Posterior distribution parameter 1, either nu1 or mean, determined by `enc_config`
            param2: list of torch.Tensor
                Posterior distribution parameter 2, either nu2 or variance, determined by `enc_config`
            total_loss: torch.Tensor
            rec_losses: list of torch.tensor
                Reconstruction loss for each dataset
            KL_loss: torch.Tensor
        """
        batch_size = self._fetch_batch_size(x)
        
        if self.mask_missing is not None:
            param1, param2 = self.encode(self.mask_missing(x), **enc_kwargs)
        else:
            param1, param2 = self.encode(x, **enc_kwargs)
        # if param1[0] is not None and param1[1] is not None:
        #     print('nu1 max:', torch.abs(param1[0]).max().item(), 'nu1 mean:', torch.abs(param1[0]).mean().item())
        #     print('nu1p max:', torch.abs(param1[1]).max().item(), 'nu1p mean:', torch.abs(param1[1]).mean().item())
        #     print('nu2 min:', torch.abs(param2[0]).min().item(), 'nu2 mean:', torch.abs(param2[0]).mean().item())
        #     print('nu2p min:', torch.abs(param2[1]).min().item(), 'nu2p mean:', torch.abs(param2[1]).mean().item())
        #     assert torch.isnan(param1[0]).sum() == 0
        
        G = self.get_G().detach() if detach_G else self.get_G()
    
        if self.KL_calc == 'derivative_gradient':
            (z_priors, z_posteriors, 
             T_priors, T_posteriors, nus) = self._sampling_gradient(G, param1, param2, 
                                                                    n_iterations=n_gibbs_iter)
            kl = KLGradient.apply(*T_priors, *T_posteriors, G, *nus)
        else:
            z_posteriors, kl = self._sampling_autograd(G, param1, param2, 
                                                       n_iterations=n_gibbs_iter)
        
        # assert torch.isnan(G).sum() == 0
        # assert torch.isnan(z_posteriors[0]).sum() == 0
        # assert torch.isnan(z_posteriors[1]).sum() == 0

        # TODO: deal with missing data
        x_rec, neg_loglike = self.decode(z_posteriors, x, **dec_kwargs) # Decoding
        # assert torch.isnan(x_rec[0][0]).sum() == 0
        # assert torch.isnan(x_rec[1][0]).sum() == 0
        
        # Total loss
        if all(xi is None for xi in x): # No rec loss
            total_loss = kl_weight * kl
            rec_loss = [0] * self.M
        else:
            rec_loss = [i.detach().mean().item() for i in neg_loglike]
            dec_rec_loss = sum(neg_loglike)
            total_loss = kl_weight * kl + dec_rec_loss.mean()
            if self.KL_calc == 'derivative_gradient':
                total_loss += RecGradient.apply(*T_posteriors, G, *nus, dec_rec_loss.detach())
            
        # These will then be used for logging only. Don't waste CUDA memory!
        z_posteriors = [i[:, -1].detach().cpu() for i in z_posteriors]
        x_rec = [i[:, -1].detach().cpu() for i in x_rec] # -1 select the last Gibbs sample
        param1 = [i.detach().cpu() if i is not None else None for i in param1]
        param2 = [i.detach().cpu() if i is not None else None for i in param2]
        results = {
            'z': z_posteriors, 'x_rec': x_rec, 'param1': param1, 'param2': param2,
            'total_loss': total_loss, 'rec_losses': rec_loss, 'KL_loss': kl.item()
        }
        return results

    
    def generate(self, n_samples, img_dims, n_gibbs_iter=15, dec_kwargs={}):
        self._batch_size = n_samples
        G = self.get_G()
        
        nones = [None] * len(self.latent_dims)
        
        if self.KL_calc == 'derivative_gradient':
            _, z_posteriors, _, _, _ = self._sampling_gradient(G, nones, nones, 
                                                               n_iterations=n_gibbs_iter)
        else:
            z_posteriors, _ = self._sampling_autograd(G, nones, nones, 
                                                       n_iterations=n_gibbs_iter)
        
        zeros = [torch.zeros(n_samples, *d) for d in img_dims]
        x_rec, _ = self.decode(z_posteriors, zeros, **dec_kwargs)
        
        z_posteriors = [i[:, -1].detach().cpu() for i in z_posteriors]
        x_rec = [i[:, -1].detach().cpu() for i in x_rec]
        results = {'z': z_posteriors, 'x_rec': x_rec}
        
        return results

    
    def _fetch_batch_size(self, x):
        if not self.batched:
            return 1
        for xi in x:
            if xi is not None:
                return xi.shape[0]
        return self._batch_size