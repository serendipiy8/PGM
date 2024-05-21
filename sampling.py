# import torch
# import abc
# from typing import Tuple
# from loss import get_score_fn
# from sde import SDE
# from solver import ReverseDiffusionPredictor, LangevinCorrector
# from utils.graph_utils import mask_adjs, mask_x
#
# # get_predictor, register_predictor = register_category("predictors")
# # get_corrector, register_corrector = register_category("correctors")
# class Predictor(abc.ABC):
#     """The abstract class for a predictor algorithm."""
#
#     def __init__(
#             self,
#             sde: SDE,
#     ):
#         super().__init__()
#         self.sde = sde
#
#     @abc.abstractmethod
#     def update_fn(
#             self, rng: torch.Generator, x: torch.Tensor, t: float, dt: float
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """One update of the predictor.
#
#         Args:
#           rng: A PyTorch random generator.
#           x: A PyTorch tensor representing the current state
#           t: A float representing the current time step.
#
#         Returns:
#           x: A PyTorch tensor of the next state.
#           x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
#         """
#         raise NotImplementedError()
#
#
# class Corrector(abc.ABC):
#     """The abstract class for a corrector algorithm."""
#
#     def __init__(
#             self,
#             sde: SDE,
#             snr: float,
#             n_steps: int,
#     ):
#         super().__init__()
#         self.sde = sde
#         self.snr = snr
#         self.n_steps = n_steps
#
#     @abc.abstractmethod
#     def update_fn(
#             self, rng: torch.Generator, x: torch.Tensor, t: float, dt: float
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """One update of the corrector.
#
#         Args:
#           rng: A PyTorch random generator.
#           x: A PyTorch tensor representing the current state
#           t: A float representing the current time step.
#
#         Returns:
#           x: A PyTorch tensor of the next state.
#           x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
#         """
#         raise NotImplementedError()
#
# @register_predictor
# class EulerMaruyamaPredictor(Predictor):
#     def __init__(self, sde):
#         super().__init__(sde)
#
#     def update_fn(
#             self, rng: torch.Generator, x: torch.Tensor, t: float, dt: float
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         z = torch.randn(x.shape, generator=rng)
#         drift, diffusion = self.sde.coefficients(x, t)
#         x_mean = x + drift * dt
#
#         if len(diffusion.shape) > 1 and diffusion.shape[-1] == diffusion.shape[-2]:
#             # if square matrix diffusion coeffs
#             diffusion_term = torch.einsum(
#                 "...ij,j,...->...i", diffusion, z, torch.sqrt(torch.abs(dt))
#             )
#         else:
#             # if scalar diffusion coeffs (i.e. no extra dims on the diffusion)
#             diffusion_term = torch.einsum(
#                 "...,...i,...->...i", diffusion, z, torch.sqrt(torch.abs(dt))
#             )
#
#         x = x_mean + diffusion_term
#         return x, x_mean
#
#
# @register_predictor
# class NonePredictor(Predictor):
#     """An empty predictor that does nothing."""
#
#     def __init__(self, sde):
#         pass
#
#     def update_fn(
#             self, rng: torch.Generator, x: torch.Tensor, t: float, dt: float
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         return x, x
#
#
# @register_corrector
# class NoneCorrector(Corrector):
#     """An empty corrector that does nothing."""
#
#     def __init__(
#             self,
#             sde: SDE,
#             snr: float,
#             n_steps: int,
#     ):
#         pass
#
#     def update_fn(
#             self, rng: torch.Generator, x: torch.Tensor, t: float, dt: float
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         return x, x
#
# # def get_pc_sampler(
# #         sde: SDE,
# #         N: int,
# #         predictor: Predictor = "EulerMaruyamaPredictor",
# #         corrector: Corrector = None,
# #         inverse_scaler=lambda x: x,
# #         snr: float = 0.2,
# #         n_steps: int = 1,
# #         denoise: bool = True,
# #         eps: float = 1e-3,
# #         return_hist=False,
# # ):
# #     predictor = get_predictor(predictor if predictor is not None else "NonePredictor")(sde)
# #     corrector = get_corrector(corrector if corrector is not None else "NoneCorrector")(
# #         sde, snr, n_steps
# #     )
# #
# #     def pc_sampler(rng, x, t0=None, tf=None):
# #         t0 = sde.t0 if t0 is None else t0
# #         tf = sde.tf if tf is None else tf
# #         t0 = torch.broadcast_to(t0, x.shape[0])
# #         tf = torch.broadcast_to(tf, x.shape[0])
# #
# #         if isinstance(sde, RSDE):
# #             tf = tf + eps
# #         else:
# #             t0 = t0 + eps
# #
# #         timesteps = torch.linspace(start=t0, end=tf, steps=N)
# #         dt = (tf - t0) / N
# #
# #         def loop_body(i, val):
# #             rng, x, x_mean, x_hist = val
# #             t = timesteps[i]
# #             rng, step_rng = torch.manual_seed().split(rng)
# #             x, x_mean = corrector.update_fn(step_rng, x, t, dt)
# #             rng, step_rng = torch.manual_seed().split(rng)
# #             x, x_mean = predictor.update_fn(step_rng, x, t, dt)
# #
# #             x_hist[i] = x
# #
# #             return rng, x, x_mean, x_hist
# #
# #         x_hist = torch.zeros((N, *x.shape))
# #
# #         _, x, x_mean, x_hist = torch.arange(N).fold((rng, x, x, x_hist), loop_body)
# #
# #         if return_hist:
# #             return (
# #                 inverse_scaler(x_mean if denoise else x),
# #                 inverse_scaler(x_hist),
# #                 timesteps,
# #             )
# #         else:
# #             return inverse_scaler(x_mean if denoise else x)
# #
# #     return pc_sampler
#
# def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None',
#                    snr=0.1, scale_eps=1.0, n_steps=1,
#                    probability_flow=False, continuous=False,
#                    denoise=True, eps=1e-3, device='cuda'):
#
#   def pc_sampler(model_x, model_adj, init_flags):
#
#     score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
#     score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)
#
#     predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
#     corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector
#
#     predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
#     corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)
#
#     predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
#     corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)
#
#     with torch.no_grad():
#       # -------- Initial sample --------
#       x = sde_x.prior_sampling(shape_x).to(device)
#       adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
#       flags = init_flags
#       x = mask_x(x, flags)
#       adj = mask_adjs(adj, flags)
#       diff_steps = sde_adj.N
#       timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
#
#       # -------- Reverse diffusion process --------
#       for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
#         t = timesteps[i]
#         vec_t = torch.ones(shape_adj[0], device=t.device) * t
#
#         _x = x
#         x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
#         adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)
#
#         _x = x
#         x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
#         adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
#       print(' ')
#       return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
#   return pc_sampler
