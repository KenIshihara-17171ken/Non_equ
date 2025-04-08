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


import os
import sys
import numpy as np
import timeit

# Ensure local imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import container
import exp_max
from ssll_kinetic.probability import log_marginal

def run(spikes, max_iter=100, mstep=True):
    """
    Runs the Expectation-Maximization (EM) algorithm to fit the state-space kinetic Ising model
    to spike data.

    :param numpy.ndarray spikes:
        A binary array of shape (T, R, N) representing spike data, where T is the number of time bins,
        R is the number of trials, and N is the number of neurons. A value of 1 indicates a spike,
        and 0 indicates no spike.
    :param int max_iter:
        The maximum number of EM iterations to perform. Default is 100.
    :param bool mstep:
        Whether to perform the maximization step (M-step) at each iteration. Default is True.

    :returns:
        container.EMData
            An EMData object containing:
              - The posterior parameter estimates (emd.theta_s).
              - The log marginal likelihood for each iteration (emd.mllk_list).
              - The sequence of state covariance matrices (emd.Q_list).
              - The total number of completed iterations (emd.iterations).
              - Timing information for the E-step, M-step, and log-likelihood calculations.
              - The Akaike Information Criterion (AIC) based on the final likelihood.
    """
    # Initialize the EMData container with the given spike data
    emd = container.EMData(spikes)

    # Compute the initial marginal log likelihood
    lmc = emd.marg_llk(emd)
    mllk = np.inf

    # Initialize iteration count
    emd.iterations = 0

    # EM loop
    while emd.iterations < max_iter:
        print(
            f"EM Iteration: {emd.iterations} - Convergence {emd.convergence:.6f} > {emd.CONVERGED:.6f}"
        )

        # E-step timing
        loop = 1
        e_step_time = timeit.timeit(lambda: exp_max.e_step(emd), number=loop)
        emd.e_step_time = e_step_time / loop

        # M-step timing (only if mstep=True)
        if mstep:
            m_step_time = timeit.timeit(lambda: exp_max.m_step(emd), number=loop)
            emd.m_step_time = m_step_time / loop

        # Update previous log likelihood
        lmp = lmc
        # Compute new log likelihood
        lmc = emd.marg_llk(emd)

        # Log-likelihood calculation timing
        llk_time = timeit.timeit(lambda: emd.marg_llk(emd), number=loop)
        emd.llk_time = llk_time / loop

        # Store marginal log likelihood and state covariance
        emd.mllk_list.append(lmc)
        emd.mllk = lmc
        emd.Q_list.append(emd.state_cov)
        emd.iterations_list.append(emd.iterations)

        # Increment iteration count
        emd.iterations += 1

        # Compute convergence based on relative change in log likelihood
        emd.convergence = (lmp - lmc) / lmp if lmp != 0 else 0

    # Compute AIC after finishing
    emd.aic = -2 * emd.mllk + 2 * emd.dim_pram
    print("Log Likelihood:", emd.mllk, "iter:", emd.iterations)
    print("emd.dim_pram:", emd.dim_pram)
    print("AIC:", emd.aic)

    return emd
