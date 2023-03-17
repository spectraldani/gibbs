import gpytorch
import torch

__all__ = ['SoR']


class SparseRegressionBase:
    def __init__(self, kernel, lik_variance, inducing_inputs, jitter=1e-7):
        self.kernel = kernel
        self.lik_variance = lik_variance
        self.inducing_inputs = inducing_inputs
        self.jitter = jitter

    def prior(self, x):
        raise NotImplementedError()

    def posterior(self, x, y):
        raise NotImplementedError()

    def training_prior(self, x):
        return self.prior(x)

    def training_posterior(self, x, y):
        return self.posterior(x, y)


class SoR(SparseRegressionBase):
    def prior(self, x):
        cov_u = self.kernel(self.inducing_inputs)
        cov_uf = self.kernel(self.inducing_inputs, x).evaluate()
        qov_f = cov_u.inv_matmul(cov_uf, cov_uf.T)
        return gpytorch.distributions.MultivariateNormal(
            torch.zeros(len(x)), qov_f + self.jitter * torch.eye(len(x))
        )

    def posterior(self, x, y):
        cov_u = self.kernel(self.inducing_inputs)
        cov_uf = self.kernel(self.inducing_inputs, x).evaluate()
        sigma = (cov_u + cov_uf @ cov_uf.T / self.lik_variance)
        return gpytorch.distributions.MultivariateNormal(
            sigma.inv_matmul(cov_uf @ y, cov_uf.T / self.lik_variance),
            sigma.inv_matmul(cov_uf, cov_uf.T) + self.jitter * torch.eye(len(x))
        )


class DTC(SoR):
    def prior(self, x):
        cov_f = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(torch.zeros(len(x)), cov_f)

    def posterior(self, x, y):
        cov_u = self.kernel(self.inducing_inputs)
        cov_uf = self.kernel(self.inducing_inputs, x).evaluate()
        cov_f = self.kernel(x)
        qov_f = cov_u.inv_matmul(cov_uf, cov_uf.T)
        sigma = (cov_u + cov_uf @ cov_uf.T / self.lik_variance)
        return gpytorch.distributions.MultivariateNormal(
            sigma.inv_matmul(cov_uf @ y, cov_uf.T / self.lik_variance),
            cov_f - qov_f + sigma.inv_matmul(cov_uf, cov_uf.T) + self.jitter * torch.eye(len(x))
        )

    def training_prior(self, x):
        return super().prior(x)

    def training_posterior(self, x, y):
        return super().posterior(x, y)


class VariationalGP(DTC):
    def prior(self, x):
        raise RuntimeError('This approximation only defines a posterior')

    def training_prior(self, x):
        raise RuntimeError('This approximation only defines a posterior')

    def training_posterior(self, x, y):
        return super().posterior(x, y)
