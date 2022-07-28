from .POISE_VAE import POISEVAE
# from .POISE_VAE_gibbs_autograd import POISEVAE as POISEVAE_Gibbs_autograd
# from .POISE_VAE_gibbs_gradient import POISEVAE as POISEVAE_Gibbs_gradient
import poisevae.utils
import poisevae.networks

# def POISEVAE_Gibbs(variant, *args, **kwargs):
#     if variant == 'autograd':
#         model = POISEVAE_Gibbs_autograd(*args, **kwargs)
#     elif variant == 'gradient':
#         model = POISEVAE_Gibbs_gradient(*args, **kwargs)
#     else: 
#         raise ValueError('variant')
#     return model