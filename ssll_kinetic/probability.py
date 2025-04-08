"""
This code implements the state-space kinetic Ising model described in:
Ken Ishihara, Hideaki Shimazaki. *State-space kinetic Ising model reveals task-dependent entropy flow in sparsely active nonequilibrium neuronal dynamics*. (2025) arXiv:2502.15440

The implementation extends existing libraries available at:
- https://github.com/christiando/ssll_lib.git
- https://github.com/shimazaki/dynamic_corr

Copyright (C) 2025  
Authors of the extensions: Ken Ishihara (KenIshihara-17171ken)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np

def log_marginal(emd):
    """
    Computes the log marginal likelihood of the data given the current state estimates.

    :param container.EMData emd:
        An EMData-like object containing:
          - sigma_f (numpy.ndarray): Filtered covariance, shape (T, N, N+1, N+1).
          - sigma_o (numpy.ndarray): One-step predicted covariance, shape (T, N, N+1, N+1).
          - sigma_o_i (numpy.ndarray): Inverses of sigma_o, shape (T, N, N+1, N+1).
          - theta_f (numpy.ndarray): Filtered parameter estimates, shape (T, N, N+1).
          - theta_o (numpy.ndarray): Predicted parameter estimates, shape (T, N, N+1).
          - FSUM (numpy.ndarray): Precomputed feature sums, shape (T, N, N+1).
          - spikes (numpy.ndarray): Spike data, shape (T, R, N).
          - T (int): Number of time bins.
          - R (int): Number of trials.
    :returns: 
        float
            The computed log marginal likelihood of the data given the current estimates.
    """
    # 1) Compute log-determinants of sigma_f and sigma_o
    _, logdet_sigma_f = np.linalg.slogdet(emd.sigma_f)
    _, logdet_sigma_o = np.linalg.slogdet(emd.sigma_o)

    # 2) Quadratic penalty term based on a = theta_f - theta_o
    a = emd.theta_f - emd.theta_o
    b = 0.5 * np.einsum(
        'tij,tij->ti',
        a,
        np.einsum('tijk,tij->tik', emd.sigma_o_i, a)
    )

    # 3) Construct feature matrix F_1 = [1, spike data]
    F_1 = np.concatenate(
        [np.ones((emd.T, emd.R, 1)), emd.spikes[:emd.T]],
        axis=2
    )

    # 4) Compute PSI = sum over trials of log(1 + exp(theta_f • F_1))
    PSI = np.sum(
        np.log(
            1 + np.exp(
                np.einsum('tij,trj->tri', emd.theta_f, F_1)
            )
        ),
        axis=1
    )

    # 5) Compute q = <theta_f, FSUM> - PSI - b
    q = np.einsum('tij,tij->ti', emd.theta_f, emd.FSUM) - PSI - b

    # 6) Combine determinant and q terms for final log marginal likelihood
    log_p = np.sum(0.5 * logdet_sigma_f - 0.5 * logdet_sigma_o + q)
    return log_p
