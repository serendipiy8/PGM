import torch
import geoopt

class MultivariateNormal:
    def __init__(self, dim, mean=None, scale=None):
        mean = torch.zeros(dim) if mean is None else mean
        scale = torch.ones(dim) if scale is None else scale
        self.dim = dim
        self.mean = mean
        self.scale = scale

    def sample(self, rng, shape):
        if self.scale.dim() == 1:
            cov = torch.diag(self.scale)
        else:
            cov = self.scale
        mvn = torch.distributions.MultivariateNormal(self.mean, cov)
        return mvn.sample(shape)

    def log_prob(self, z):
        if self.scale.dim() == 1:
            cov = torch.diag(self.scale)
        else:
            cov = self.scale
        mvn = torch.distributions.MultivariateNormal(self.mean, cov)
        return mvn.log_prob(z)

    def grad_U(self, x):
        return x / (self.scale ** 2)

class WrapNormDistribution:
    def __init__(self, manifold, scale=1.0, mean=None):
        self.manifold = manifold
        if mean is None:
            mean = self.manifold.identity
        self.mean = mean
        self.scale = (
            torch.ones_like(torch.tensor(mean.shape)) * scale
            if isinstance(scale, float)
            else torch.tensor(scale)
        )

    def sample(self, rng, shape):
        mean = self.mean[None, ...]
        tangent_vec = self.manifold.random_normal_tangent(
            rng, self.manifold.identity, torch.prod(torch.tensor(shape))
        )[1]
        tangent_vec *= self.scale
        tangent_vec = self.manifold.transp(self.mean, tangent_vec)
        return self.manifold.expmap(self.mean, tangent_vec)

    def log_prob(self, z):
        tangent_vec = self.manifold.logmap(self.mean, z)
        tangent_vec = self.manifold.transpback(self.mean, tangent_vec)
        zero = torch.zeros(self.manifold.dim)
        if self.scale.shape[-1] == self.manifold.dim:  # poincare
            scale = self.scale
        else:  # hyperboloid
            scale = self.scale[..., 1:]
        mvn = MultivariateNormal(zero, scale)
        norm_pdf = mvn.log_prob(tangent_vec)
        logdetexp = self.manifold.logdetexp(self.mean, z)
        return norm_pdf - logdetexp

    def grad_U(self, x):
        def U(x):
            sq_dist = self.manifold.dist(self.mean, x) ** 2
            res = 0.5 * sq_dist / (self.scale[0] ** 2)  # scale must be isotropic
            logdetexp = self.manifold.logdetexp(self.mean, x)
            return res + logdetexp

        return self.manifold.transp(self.mean, self.manifold.grad(U)(x))
