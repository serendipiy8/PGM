import abc

import geoopt.samplers.base
import torch
import numpy as np
from utils.distribution import WrapNormDistribution,MultivariateNormal
# from sampling import get_pc_sampler

# class SDE(abc.ABC):
#     """SDE abstract class"""
#
#     def __init__(self,N,beta_schedule):
#         """Construct an SDE.
#         Args:
#             N:number of discretization time steps.
#             beta_schedule:beta equation.
#         """
#
#         super().__init__()
#         self.N=N
#
#         self.beta_schedule=beta_schedule
#         self.tf=beta_schedule.tf
#         self.t0=beta_schedule.t0
#
#     @property
#     @abc.abstractmethod
#     def T(self):
#         """End time of the SDE."""
#         pass
#
#     @abc.abstractmethod
#     def drift(self,x,t):
#         """The drift coefficients at (x, t)."""
#         pass
#
#     @abc.abstractmethod
#     def diffusion(self,x,t):
#         """The diffusion coefficients at (x, t)."""
#         pass
#
#     @abc.abstractmethod
#     def sde(self,x,t):
#         """Construct SDE state in time t."""
#         pass
#
#
#
#     def marginal_prob(self,x,t):
#         """Parameters to determine the marginal distribution of the SDE,$p_t(x)$."""
#         pass
#
#     def marginal_log_prob(self,x0,x,t):
#         """Compute the log marginal distribution of the SDE, $log p_t(x | x_0 = 0)$."""
#         pass
#
#     def grade_marginal_log_prob(self,x0,x,t):
#         """Compute the log marginal distribution and its gradient."""
#         pass
#
#
#     def sample_limiting_distribution(self,manifold,shape):
#         """Generate one example from the prior distribution,$p_T(x)$."""
#         pass
#
#     def limiting_distribution_logp(self,z):
#         """Computer log-density of the prior distribution.
#         Useful for computing the log-likelihood via probability flow ODE.
#         Args:
#             z:latent code
#         Returns:
#             log probability density
#         """
#         pass
#
#     def discretize(self,x,t):
#         """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
#
#            Useful for reverse diffusion sampling and probability flow sampling.
#            Defaults to Euler-Maruyama discretization.
#
#             Returns:
#                 f, G - the discretised SDE drift and diffusion coefficients
#         """
#         dt=1/self.N
#         drift,diffusion=self.drift(x,t)
#         f = drift * dt
#         G = diffusion * torch.sqrt(torch.tensor(dt,device=t.device))
#         return f, G
#
#     def reverse(self,score_fn,probability_flow=False):
#         """Reverse time SDE
#         Args:
#             score_fn:A time-dependent score-based model that takes x and t and returns the score.
#             probalility_flow:If 'True',create the reverse-time ODE used for probability flow sampling.
#         """
#
#         N=self.N
#         T=self.T
#         sde_fn=self.sde
#         discretize_fn=self.discretize
#
#         #-----------the class of the reverse-time SDE-----------
#         class RSDE(self.__class__):
#             def __init__(self):
#                 self.N = N
#                 self.probability_flow = probability_flow
#
#             @property
#             def T(self):
#                 return T
#
#             def sde(self,feature,x,flags,t,is_adj=True):
#                 drift,diffusion=sde_fn(x,t) if is_adj else sde_fn(feature,t)
#                 score=score_fn(feature,x,flags,t)
#                 drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
#                 # -------- Set the diffusion function to zero for ODEs. --------
#                 diffusion = 0. if self.probability_flow else diffusion
#                 return drift, diffusion
#
#             def discretize(self,feature,x,flags,t,is_adj=True):
#                 """Create discretized iteration rules for the reverse diffusion sampler."""
#                 f, G = discretize_fn(x, t) if is_adj else discretize_fn(feature, t)
#                 score = score_fn(feature, x, flags, t)
#                 rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
#                 rev_G = torch.zeros_like(G) if self.probability_flow else G
#                 return rev_f, rev_G
#
#         return RSDE()
#
#
#     def reparametrise_score_fn(self,score_fn,*args):
#         pass
#
#     def probability_ode(self,score_fn):
#         pass
#
#
# class Langevin(SDE):
#     """Construct Langevin dynamics on a manifold"""
#
#     def __init__(self,beta_schedule,manifold, ref_scale=0.5,ref_mean=None,N=1000):
#         super().__init__(N=N,beta_schedule=beta_schedule)
#         self.manifold=manifold
#         self.limiting=WrapNormDistribution(manifold,scale=ref_scale,mean=ref_mean)
#
#     def drift(self,x,t):
#         """dX_t =-0.5 beta(t) grad U(X_t)dt + sqrt(beta(t)) dB_t"""
#
#         def fixed_grad(grad):
#             grad[torch.isnan(grad)| torch.isinf(grad)]=0
#             return grad
#
#         def drift_fn(x):
#             grad_U=-0.5* fixed_grad(self.limiting.grad_U(x))
#             return grad_U
#
#         beta_t=self.beta_schedule.beta_t(t)
#         drift=beta_t*drift_fn(x)
#
#         return drift
#
#     def diffusion(self,x,t):
#         beta_t = self.beta_schedule.beta_t(t)
#         diffusion_coef = torch.sqrt(beta_t)
#         return diffusion_coef
#
#     def marginal_prob(self,x,t):
#         log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
#         mean_coeff = torch.exp(log_mean_coeff[:, None, None]) * x
#         std = torch.sqrt(1 - torch.exp(2.0 * log_mean_coeff))
#         return mean_coeff, std
#
#     def marginal_sample(self,rng,x,t,return_hist=False):
#         scaled_t=self.beta_schedule.rescale_t(t)
#
#         #Generate tangent vector
#         tangent_vector=torch.sqrt(scaled_t)*torch.randn(x.shape,device=x.device)
#
#         #Perform the random walk and use exponential map to tangent vector
#         out=self.manifold.expmap(x,tangent_vector)
#
#         if return_hist or out is None:
#             sampler = get_pc_sampler(
#                 self, self.N, predictor="GRW", return_hist=return_hist
#             )
#             out = sampler(rng, x, tf=t)
#         return out
#
#
# #-----------VP SDE-----------
# class VPSDE(Langevin):
#     """Construct a Variance Exploding SDE
#     Args:
#
#     """
#     def __init__(self,beta_schedule,N=1000,manifold=None,ref_scale=0.5,ref_mean=None):
#         super().__init__(beta_schedule,manifold,N)
#         self.manifold = manifold
#         self.limiting = WrapNormDistribution(manifold, scale=ref_scale, mean=ref_mean)
#         self.beta_schedule = beta_schedule
#
#         self.tf = beta_schedule.tf
#         self.t0 = beta_schedule.t0
#         self.N=N
#
#
#     @property
#     def T(self):
#         return 1
#
#     def sde(self,x,t):
#         beta_t=self.beta_schedule.beta_t(t)
#         drift = -0.5 * beta_t[:, None, None] * x
#         diffusion = torch.sqrt(beta_t)
#         return drift, diffusion
#
#     #--------- mean,std of the perturbation kernel ---------
#     def marginal_prob(self,x,t):
#         log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
#         mean = torch.exp(log_mean_coeff[:, None, None]) * x
#         std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
#         return mean, std
#
#     def marginal_sample(self, rng, x, t, return_hist=False):
#         mean = super().marginal_sample(rng, x, t, return_hist)  # 调用父类的方法
#         return mean
#
#     def marginal_sample_sym(self,rng,x,t,return_hist=False):
#         x=self.marginal_sample(rng,x,t).triu(1)
#         return x + x.transpose(-1,-2)
#
#     def grad_marginal_log_prob(self, x0, x, t):
#         mean, std = self.marginal_prob(x0, t)
#         std = std.unsqueeze(-1)
#         logp = -0.5 * (torch.log(2 * torch.pi) + torch.log(std ** 2) + ((x - mean) ** 2) / (std ** 2))
#         score = -1 / (std ** 2) * (x - mean)
#         return logp, score
#
#     def prior_logp(self,z):
#         shape = z.shape
#         N = np.prod(shape[1:])
#         logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
#         return logps
#
#     def discretize(self,x,t):
#         """DDPM discretization."""
#         timestep = (t * (self.N - 1) / self.T).long()
#         beta = self.discrete_betas.to(x.device)[timestep]
#         alpha = self.alphas.to(x.device)[timestep]
#         sqrt_beta = torch.sqrt(beta)
#         f = torch.sqrt(alpha)[:, None, None] * x - x
#         G = sqrt_beta
#         return f, G
#
#     def transition(self,x,t,dt):
#         #------- negative timrstep dt -------
#         log_mean_coeff = self.beta_schedule.log_mean_coeff(t)
#         mean = torch.exp(-log_mean_coeff[:, None, None]) * x
#         std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
#         return mean, std
#
# class VESDE(Langevin):
#     pass


#-----------VE SDE-----------
# class VESDE(Langevin):
#     def __init__(self,beta_schedule,N,manifold=None):
#         """Construct a Variance Exploding SDE."""
#         super().__init__(beta_schedule,manifold,N)
#
#         self.limiting = MultivariateNormal(dim=manifold)
#         self.beta_schedule = beta_schedule
#
#         self.tf = beta_schedule.tf
#         self.t0 = beta_schedule.t0
#         self.N = N
#
#     @property
#     def T(self):
#         return 1
#
#     def sde(self,x,t):
#         sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
#         drift=torch.zeros_like(x)
#         diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
#                                                     device=t.device))
#         return drift, diffusion
#
#     def marginal_prob(self,x,t):
#         std

import abc
import torch
import numpy as np


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.
    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.
    Useful for computing the log-likelihood via probability flow ODE.
    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.
    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)
    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # -------- Build the class for reverse-time SDE --------
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, feature, x, flags, t, is_adj=True):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t) if is_adj else sde_fn(feature, t)
        score = score_fn(feature, x, flags, t)
        drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # -------- Set the diffusion function to zero for ODEs. --------
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, feature, x, flags, t, is_adj=True):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t) if is_adj else discretize_fn(feature, t)
        score = score_fn(feature, x, flags, t)
        rev_f = f - G[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  # -------- mean, std of the perturbation kernel --------
  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1)
    return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None] * x - x
    G = sqrt_beta
    return f, G

  def transition(self, x, t, dt):
    # -------- negative timestep dt --------
    log_mean_coeff = 0.25 * dt * (2*self.beta_0 + (2*t + dt)*(self.beta_1 - self.beta_0) )
    mean = torch.exp(-log_mean_coeff[:, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1)
    x = x + x.transpose(-1,-2)
    return x

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G

  def transition(self, x, t, dt):
    # -------- negative timestep dt --------
    std = torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** t) - \
          torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt))
    std = torch.sqrt(std)
    mean = x
    return mean, std


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.
    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_sampling_sym(self, shape):
    x = torch.randn(*shape).triu(1)
    return x + x.transpose(-1,-2)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
