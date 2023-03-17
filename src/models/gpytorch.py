import gpytorch

__all__ = ['GPySparseRegressionGP']


class GPySparseRegressionGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, pseudo_inputs=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if pseudo_inputs is not None:
            self.covar_module = gpytorch.kernels.InducingPointKernel(
                self.base_covar_module,
                inducing_points=pseudo_inputs,
                likelihood=likelihood
            )
        else:
            self.covar_module = self.base_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def model_builder(self):
        init_args = dict(
            train_x=self.train_inputs[0], train_y=self.train_targets,
            pseudo_inputs=(
                None if not isinstance(self.covar_module, gpytorch.kernels.InducingPointKernel)
                else self.covar_module.inducing_points
            )
        )
        return init_args, self.state_dict()

    @classmethod
    def build_model(cls, init_args, state_dict):
        model = cls(**init_args)
        model.load_state_dict(state_dict)
        return model
