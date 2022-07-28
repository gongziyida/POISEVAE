import torch

def covar(T1, T2=None):
    """
    Parameters
    ----------
    T1, T2: torch.Tensor
        Shape: (batch, n_iter, latent_dim * 2)
        If T2 is given, cross covariance is calculated
    """
    n = T1.shape[1]
    n = n - 1 if n > 1 else 1
    T1_diff = T1 - T1.mean(1, keepdim=True)
    if T2 is not None:
        T2_diff = T2 - T2.mean(1, keepdim=True)
        cov = torch.bmm(T1_diff.transpose(1, 2), T2_diff) / n
    else:
        cov = torch.bmm(T1_diff.transpose(1, 2), T1_diff) / n
    return cov

class KLGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T_p, Tp_p, T_q, Tp_q, G, nu, nup):
        """
        Parameters
        ----------
        T_p, Tp_p, T_q, Tp_q, nu, nup: torch.Tensor
            Shape: (batch, n_iter, latent_dim * 2)
        G: torch.Tensor
            Shape: (batch, latent_dim * 2, latent_dim * 2)
        """
        ctx.save_for_backward(nu, nup)
        ctx.T_p, ctx.Tp_p, ctx.T_q, ctx.Tp_q = T_p, Tp_p, T_q, Tp_q
        return torch.tensor([0.0], requires_grad=True, device=G.device, dtype=G.dtype)

    @staticmethod
    def backward(ctx, w):
        with torch.no_grad():
            T_p, Tp_p, T_q, Tp_q = ctx.T_p, ctx.Tp_p, ctx.T_q, ctx.Tp_q
            nu, nup = ctx.saved_tensors 
            
            covT = covar(T_q)
            covTp = covar(Tp_q)
            covTTp = covar(T_q, Tp_q)
            
            TTp_p = (T_p.unsqueeze(-1) * Tp_p.unsqueeze(-2))
            TTp_q = (T_q.unsqueeze(-1) * Tp_q.unsqueeze(-2))
            nuT = (nu.unsqueeze(1) * T_q).sum(-1, keepdim=True).unsqueeze(-1)
            nupTp = (nup.unsqueeze(1) * Tp_q).sum(-1, keepdim=True).unsqueeze(-1)
            
            dG = (TTp_q * (nuT + nupTp)).mean(1) - TTp_q.mean(1) * (nuT + nupTp).mean(1) \
               + TTp_p.mean(1) - TTp_q.mean(1)
            
            dnu = torch.bmm(nu.unsqueeze(1), covT).squeeze(1) + \
                  torch.bmm(covTTp, nup.unsqueeze(-1)).squeeze(-1)
            dnup = torch.bmm(nup.unsqueeze(1), covTp).squeeze(1) + \
                   torch.bmm(covTTp.transpose(-2, -1), nu.unsqueeze(-1)).squeeze(-1)
            
            print('dG11 KL', torch.abs(dG[:, :dG.shape[1]//2, :dG.shape[2]//2]).mean().item(), 
                  'dG12 KL', torch.abs(dG[:, :dG.shape[1]//2, dG.shape[2]//2:]).mean().item())
            print('dG21 KL', torch.abs(dG[:, dG.shape[1]//2:, :dG.shape[2]//2]).mean().item(), 
                  'dG22 KL', torch.abs(dG[:, dG.shape[1]//2:, dG.shape[2]//2:]).mean().item())
            print('dnu1 KL', torch.abs(dnu[..., :dnu.shape[-1]//2]).mean().item(), 
                  'dnu2 KL', torch.abs(dnu[..., dnu.shape[-1]//2:]).mean().item())
            print('dnu1p KL', torch.abs(dnup[..., :dnup.shape[-1]//2]).mean().item(), 
                  'dnu2p KL', torch.abs(dnup[..., dnup.shape[-1]//2:]).mean().item())
            
            return None, None, None, None, w * dG, w * dnu, w * dnup
            # return None, None, None, None, None, None, None
        
        
class RecGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T_q, Tp_q, G, nu, nup, loglike):
        """
        Parameters
        ----------
        T_q, Tp_q, nu, nup: torch.Tensor
            Shape: (batch, n_iter, latent_dim * 2)
        G: torch.Tensor
            Shape: (batch, latent_dim * 2, latent_dim * 2)
        loglike: torch.Tensor
            Shape: (batch, n_iter); correspond to the Gibbs sample used for reconstruction
        """
        ctx.save_for_backward(nu, nup)
        ctx.loglike, ctx.T_q, ctx.Tp_q = loglike.unsqueeze(-1), T_q, Tp_q
        return torch.tensor([0.0], requires_grad=True, device=G.device, dtype=G.dtype)
    @staticmethod
    def backward(ctx, w):
        with torch.no_grad():
            # Notice that T's shape (batch, n_iter, latent_dim)
            nu, nup = ctx.saved_tensors 
            loglike, T_q, Tp_q = ctx.loglike, ctx.T_q, ctx.Tp_q
            dG = (T_q.unsqueeze(-1) * Tp_q.unsqueeze(-2) * loglike.unsqueeze(-1)).mean(1) - \
                 (T_q.unsqueeze(-1) * Tp_q.unsqueeze(-2)).mean(1) * loglike.unsqueeze(-1).mean(1)
            dnu = (T_q * loglike).mean(1) - T_q.mean(1) * loglike.mean(1)
            dnup = (Tp_q * loglike).mean(1) - Tp_q.mean(1) * loglike.mean(1)
            
            print('dG11 Rec', torch.abs(dG[:, :dG.shape[1]//2, :dG.shape[2]//2]).mean().item(), 
                  'dG12 Rec', torch.abs(dG[:, :dG.shape[1]//2, dG.shape[2]//2:]).mean().item())
            print('dG21 Rec', torch.abs(dG[:, dG.shape[1]//2:, :dG.shape[2]//2]).mean().item(), 
                  'dG22 Rec', torch.abs(dG[:, dG.shape[1]//2:, dG.shape[2]//2:]).mean().item())
            print('dnu1 Rec', torch.abs(dnu[..., :dnu.shape[-1]//2]).mean().item(), 
                  'dnu2 Rec', torch.abs(dnu[..., dnu.shape[-1]//2:]).mean().item())
            print('dnu1p Rec', torch.abs(dnup[..., :dnup.shape[-1]//2]).mean().item(), 
                  'dnu2p Rec', torch.abs(dnup[..., dnup.shape[-1]//2:]).mean().item())
            
            return None, None, w * dG, w * dnu, w * dnup, None
            # return None, None, None, None, None, None