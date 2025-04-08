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
import random

def get_THETA_gaussian_process(T, N, mu=1.0, sigma=5.0, alpha=1.0):
    """
    Generates a 3D array of shape (T, N, N+1) containing time-varying parameters
    for a state-space model. Each neuron i's parameters are drawn from
    `generate_thetas(...)`. The complexity is O(N) because we loop over N neurons.

    :param int T:
        The number of time bins.
    :param int N:
        The number of neurons.
    :param float mu:
        The mean of the parameters. Default is 1.0.
    :param float sigma:
        The standard deviation controlling how parameters change over time. Default is 5.0.
    :param float alpha:
        A scalar controlling the correlation over time in the Gaussian process. Default is 1.0.

    :returns:
        numpy.ndarray
            A 3D array of shape (T, N, N+1) containing the time-varying parameters.
            - Dimension N+1 corresponds to 1 field parameter + N coupling parameters.
    """
    THETA = np.zeros((T, N, N+1))
    for i in range(N):
        THETA[:, i, :] = generate_thetas(T, D=N+1, mu=mu, sigma=sigma, alpha=alpha)
    return THETA

def generate_thetas(T, D, mu=1.0, sigma=0.5, alpha=1.0):
    """
    Generates a matrix of Gaussian process-based parameters for a state-space model.
    This function uses a Cholesky decomposition to sample from a correlated Gaussian
    process over T time bins and D dimensions.

    :param int T:
        Number of time bins.
    :param int D:
        Dimensionality of the model (e.g., N+1 if N is the number of neurons).
    :param float mu:
        Mean of the Gaussian process. Default is 1.0.
    :param float sigma:
        Standard deviation controlling how quickly parameters vary over time. Default is 0.5.
    :param float alpha:
        Scalar controlling the overall scale of the covariance kernel. Default is 1.0.

    :returns:
        numpy.ndarray
            A 2D array of shape (T, D) with the generated parameters for each time bin and dimension.
    """
    MU = np.full((T, D), mu)
    X = np.arange(T)
    # Build the covariance matrix K based on a Gaussian kernel
    K = (1.0 / alpha) * np.exp(-((X[:, None] - X[None, :]) ** 2) / (2.0 * sigma**2))

    # Cholesky decomposition for sampling
    L = np.linalg.cholesky(K + 1e-13 * np.eye(T))
    THETA = mu + np.dot(L, np.random.randn(T, D))
    return THETA

def get_S_function(T, R, N, THETA):
    """
    Generates spike data for T time bins, R trials, and N neurons using
    a state-space Ising-like model with given parameters THETA.

    The returned spike array has shape (T+1, R, N):
      - The first index (t=0) is an initial state, generated randomly.
      - Subsequent time steps (t=1..T) are generated according to the model.

    :param int T:
        Number of time bins (excluding the initial state at t=0).
    :param int R:
        Number of trials (or runs).
    :param int N:
        Number of neurons.
    :param numpy.ndarray THETA:
        The parameter array of shape (T, N, N+1) used to generate spikes.
        - THETA[t, n, 0] is the field parameter for neuron n at time t.
        - THETA[t, n, 1..N] are the coupling parameters for neuron n at time t.

    :returns:
        numpy.ndarray
            A binary 3D array of shape (T+1, R, N), where spikes[t, r, n]
            indicates whether neuron n fired in trial r at time t.
    """
    psi = np.zeros((T, R, N))
    spikes = np.zeros((T + 1, R, N))
    rand_numbers = np.random.rand(T + 1, R, N)

    # Initialize spikes at time t=0
    spikes[0] = (rand_numbers[0] >= 0.8).astype(int)

    # Generate spike data for times t=1..T
    for t in range(1, T + 1):
        # F_psi: shape (R, 1+N) -> includes 1 for field, plus previous spikes for coupling
        F_psi = np.concatenate([np.ones((R, 1)), spikes[t - 1]], axis=1)

        # psi[t-1] is the log-partition function for logistic regression
        psi[t - 1] = np.log(1 + np.exp(THETA[t - 1] @ F_psi.T)).T

        # p1 is the probability of spiking
        p1 = np.exp(THETA[t - 1] @ F_psi.T - psi[t - 1].T).T

        # Compare p1 to random draws to decide spikes
        spikes[t] = (p1 >= rand_numbers[t]).astype(int)

    return spikes

def shuffle_spikes(spikes):
    """
    Randomly permutes the trial dimension for each neuron, effectively reassigning
    the order of trials per neuron. This can be used to break trial-to-trial
    correlations while preserving within-trial structure.

    :param numpy.ndarray spikes:
        A binary spike array of shape (time, trials, neurons).

    :returns:
        numpy.ndarray
            A new array of the same shape, where the trials have been permuted
            independently for each neuron.
    """
    T, R, N = spikes.shape
    shuffled_spikes = np.zeros(spikes.shape)
    np.random.seed(1)  # For reproducibility

    for n in range(N):
        # Permute trial indices
        r_idx = np.random.permutation(np.arange(R))
        # Reassign the trials for neuron n
        shuffled_spikes[:, :, n] = spikes[:, r_idx, n]

    return shuffled_spikes

def get_THETA_gaussian_process_fixed_seed_per_neuron(T, N, mu=1.0, sigma=5.0, alpha=1.0, base_seed=100):
    """
    Generates time-varying parameters for each neuron using a fixed but distinct
    random seed for each neuron. This can help control randomness for reproducibility
    across neurons.

    :param int T:
        Number of time bins.
    :param int N:
        Number of neurons.
    :param float mu:
        Mean of the parameters. Default is 1.0.
    :param float sigma:
        Standard deviation controlling how parameters change over time. Default is 5.0.
    :param float alpha:
        A scalar controlling the correlation over time in the Gaussian process. Default is 1.0.
    :param int base_seed:
        Base random seed to be offset by neuron index.

    :returns:
        numpy.ndarray
            A 3D array of shape (T, N, N+1) containing the time-varying parameters,
            generated by a per-neuron fixed seed approach.
    """
    THETA = np.zeros((T, N, N+1))
    for i in range(N):
        rng = np.random.RandomState(base_seed + i)
        THETA[:, i, :] = generate_thetas_with_rng(T, D=N+1, mu=mu, sigma=sigma, alpha=alpha, rng=rng)
    return THETA

def generate_thetas_with_rng(T, D, mu, sigma, alpha, rng):
    """
    Similar to generate_thetas, but uses a specified random generator (rng)
    to enable fixed-seed reproducibility. This allows controlling randomness
    per neuron or per experimental condition.

    :param int T:
        Number of time bins.
    :param int D:
        Dimensionality of the model (e.g., N+1 if N is the number of neurons).
    :param float mu:
        Mean of the Gaussian process.
    :param float sigma:
        Standard deviation controlling how parameters vary over time.
    :param float alpha:
        Scalar controlling the overall scale of the covariance kernel.
    :param numpy.random.RandomState rng:
        A NumPy RandomState (or similar) to manage random draws.

    :returns:
        numpy.ndarray
            A 2D array of shape (T, D) with the generated parameters for each time bin and dimension.
    """
    X = np.arange(T)
    K = (1.0 / alpha) * np.exp(-((X[:, None] - X[None, :]) ** 2) / (2.0 * sigma**2))
    L = np.linalg.cholesky(K + 1e-13 * np.eye(T))
    THETA = mu + np.dot(L, rng.randn(T, D))
    return THETA
