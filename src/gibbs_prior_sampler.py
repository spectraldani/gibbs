from typing import Callable, Optional, Tuple, List

import torch

__all__ = ['ConditionalDistribution', 'gibbs_transition', 'run_gibbs_chain']

ConditionalDistribution = Callable[[torch.Tensor], torch.distributions.Distribution]


def gibbs_transition(
        theta_t: torch.Tensor, likelihood: ConditionalDistribution, approximate_posterior: ConditionalDistribution
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sample from likelihood
    p_y_t = likelihood(theta_t)
    y_t = p_y_t.sample()
    # Sample from approximation
    q_thetatp1 = approximate_posterior(y_t)
    theta_tp1 = q_thetatp1.sample()
    return theta_tp1, y_t


def run_gibbs_chain(
        steps: int, initial_sample: torch.Tensor,
        likelihood: ConditionalDistribution,
        approximate_posterior: ConditionalDistribution,
        burn_in: int = 0,
        monitor: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :return (samples from gibbs prior, samples from likelihood)
    """
    # noinspection PyTypeChecker
    samples_f: List[torch.Tensor] = [None] * steps
    # noinspection PyTypeChecker
    samples_y: List[torch.Tensor] = [None] * steps

    with torch.no_grad():
        f_t = initial_sample
        for t in range(burn_in):
            f_t, y_t = gibbs_transition(f_t, likelihood, approximate_posterior)
        for t in range(steps):
            f_tp1, y_t = gibbs_transition(f_t, likelihood, approximate_posterior)
            samples_f[t] = f_t
            samples_y[t] = y_t
            f_t = f_tp1
            if monitor is not None and t % 1000 == 0:
                monitor(t, samples_f, samples_y)
    return torch.stack(samples_f), torch.stack(samples_y)
