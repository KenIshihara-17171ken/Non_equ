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
from joblib import Parallel, delayed
from numba import njit

MAX_GA_ITERATIONS = 5000
GA_CONVERGENCE = 1e-4

def e_step(emd):
    """
    Performs the E-step of the EM algorithm, combining a filtering and smoothing procedure.

    :param emd: container.EMData
        The data structure holding all necessary model parameters and spike data.
    :returns: None
    """
    filter_function(emd)
    # If parallel filtering is preferred, uncomment:
    # filter_function_Parallel(emd)

    smoothing_function(emd)
    # If parallel smoothing is preferred, uncomment:
    # smoothing_function_Parallel(emd)

def m_step(emd):
    """
    Performs the M-step of the EM algorithm, updating state covariance and initial covariance matrices.

    :param emd: container.EMData
        The data structure holding the current state of model parameters.
    :returns: None
    """
    # Optionally choose a Q update method:
    # get_Q(emd)
    # get_scalar_q(emd)
    get_diagonal_Q(emd)
    get_Sigma(emd)
    # get_init_theta(emd)

def get_init_theta(emd):
    """
    Updates the initial theta estimate to the smoothed estimate at time 0.

    :param emd: container.EMData
        The data structure holding the smoothed parameter estimates.
    :returns: numpy.ndarray
        The updated initial theta of shape (N, N+1).
    """
    emd.init_theta = np.zeros((emd.N, emd.N+1))
    for i in range(emd.N):
        emd.init_theta[i] = emd.theta_s[0, i]
    return emd.init_theta

def get_Sigma(emd):
    """
    Updates the initial covariance matrices based on the smoothed estimates at time 0.

    :param emd: container.EMData
        The data structure containing smoothed parameters and covariances.
    :returns: numpy.ndarray
        The updated init_cov of shape (N, N+1, N+1).
    """
    emd.init_cov = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        diff = emd.theta_s[0, i] - emd.init_theta[i]
        emd.init_cov[i] = emd.sigma_s[0, i] + np.outer(diff, diff)
    return emd.init_cov

def get_Q(emd):
    """
    Computes a fully dense state covariance matrix Q for each neuron.

    :param emd: container.EMData
        The data structure with smoothed parameters (theta_s, sigma_s).
    :returns: numpy.ndarray
        The updated state_cov of shape (N, N+1, N+1).
    """
    Q = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        tmp = np.zeros((emd.N+1, emd.N+1))
        for t in range(1, emd.T):
            tmp += (
                np.outer(emd.theta_s[t, i], emd.theta_s[t, i])
                + emd.sigma_s[t, i]
                - np.outer(emd.theta_s[t-1, i], emd.theta_s[t, i])
                - emd.lag_one_covariance[t-1, i]
                - np.outer(emd.theta_s[t, i], emd.theta_s[t-1, i])
                - emd.lag_one_covariance[t-1, i].T
                + np.outer(emd.theta_s[t-1, i], emd.theta_s[t-1, i])
                + emd.sigma_s[t-1, i]
            )
        Q[i] = tmp / (emd.T - 1)
        # Symmetrize
        Q[i] = (Q[i] + Q[i].T) / 2
    emd.state_cov = Q
    # Update estimated dimension of parameters (example formula)
    emd.dim_pram = (
        ((emd.N+1)*(emd.N+1 - 1)/2 + (emd.N+1)
         + (emd.N+1)*(emd.N+1 - 1)/2) * emd.N
    )
    return emd.state_cov

def get_scalar_q(emd):
    """
    Computes a single scalar q (isotropic) for each neuron and updates the state covariance.

    :param emd: container.EMData
        The data structure with smoothed parameters (theta_s, sigma_s).
    :returns: numpy.ndarray
        The updated state_cov of shape (N, N+1, N+1),
        where each neuron's Q is a scalar times the identity matrix.
    """
    Q = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        tmp = np.zeros((emd.N+1, emd.N+1))
        for t in range(1, emd.T):
            tmp += (
                np.outer(emd.theta_s[t, i], emd.theta_s[t, i])
                + emd.sigma_s[t, i]
                - np.outer(emd.theta_s[t-1, i], emd.theta_s[t, i])
                - emd.lag_one_covariance[t-1, i]
                - np.outer(emd.theta_s[t, i], emd.theta_s[t-1, i])
                - emd.lag_one_covariance[t-1, i].T
                + np.outer(emd.theta_s[t-1, i], emd.theta_s[t-1, i])
                + emd.sigma_s[t-1, i]
            )
        Q[i] = tmp / (emd.T - 1)
        Q[i] = (Q[i] + Q[i].T) / 2

        trace_Q = np.trace(Q[i])
        qi = trace_Q / (emd.N + 1)
        emd.state_cov[i] = qi * np.eye(emd.N + 1)

    emd.dim_pram = (
        (1 + (emd.N+1) + (emd.N+1)*(emd.N+1 - 1)/2) * emd.N
    )
    return emd.state_cov

def get_diagonal_Q(emd):
    """
    Computes a diagonal state covariance matrix Q for each neuron by averaging each diagonal element.

    :param emd: container.EMData
        The data structure with smoothed parameters (theta_s, sigma_s).
    :returns: numpy.ndarray
        The updated state_cov of shape (N, N+1, N+1),
        where each neuron's Q is diagonal.
    """
    Q = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        tmp = np.zeros((emd.N+1, emd.N+1))
        for t in range(1, emd.T):
            tmp += (
                np.outer(emd.theta_s[t, i], emd.theta_s[t, i])
                + emd.sigma_s[t, i]
                - np.outer(emd.theta_s[t-1, i], emd.theta_s[t, i])
                - emd.lag_one_covariance[t-1, i]
                - np.outer(emd.theta_s[t, i], emd.theta_s[t-1, i])
                - emd.lag_one_covariance[t-1, i].T
                + np.outer(emd.theta_s[t-1, i], emd.theta_s[t-1, i])
                + emd.sigma_s[t-1, i]
            )
        diag_values = np.diag(tmp) / (emd.T - 1)
        Q[i] = np.diag(diag_values)

    emd.state_cov = Q
    emd.dim_pram = (
        ((emd.N+1) + (emd.N+1) + (emd.N+1)*(emd.N+1 - 1)/2) * emd.N
    )
    return emd.state_cov

def cal_eta_G(theta, R, spikes_t):
    """
    Computes the first derivative (eta) and the Fisher information matrix (G) using Einstein summation.

    :param numpy.ndarray theta:
        Current parameter estimates of shape (N, N+1).
    :param int R:
        Number of trials.
    :param numpy.ndarray spikes_t:
        Spike data at time t of shape (R, N).

    :returns: tuple
        (eta, G)
        eta: numpy.ndarray of shape (N, N+1)
        G: numpy.ndarray of shape (N, N+1, N+1)
    """
    F1 = np.concatenate([np.ones((R, 1)), spikes_t], axis=1)
    # r: (R, N)
    r = 1 / (1 + np.exp(-np.einsum('ij,kj->ki', theta, F1)))
    # eta: (N, N+1)
    eta = np.einsum('ki,kj->ij', r, F1)
    # G: (N, N+1, N+1)
    G = np.einsum('ki,kjl->ijl', r * (1 - r), np.einsum('kj,kl->kjl', F1, F1))
    return eta, G

def filter_function(emd):
    """
    Filters the parameter estimates forward in time using Newton-Raphson updates (Einstein summation).
    Complexity: O(N * R * T)

    :param emd: container.EMData
        The data structure with all necessary parameters and spike data.
    :returns: tuple
        Updated (theta_f, sigma_f, sigma_f_i, sigma_o, sigma_o_i).
    """
    for t in range(emd.T):
        if t == 0:
            emd.theta_o[0] = emd.init_theta
            emd.sigma_o[0] = emd.init_cov
            emd.sigma_o_i[0] = np.linalg.inv(emd.sigma_o[0])
        else:
            emd.theta_o[t] = emd.theta_f[t - 1]
            emd.sigma_o[t] = emd.sigma_f[t - 1] + emd.state_cov
            emd.sigma_o_i[t] = np.linalg.inv(emd.sigma_o[t])

        max_dlpo = np.inf
        iterations = 0

        while max_dlpo > GA_CONVERGENCE:
            eta, G = cal_eta_G(emd.theta_f[t], emd.R, emd.spikes[t])
            dlpo = - (emd.FSUM[t] - eta) + np.einsum(
                'ijk,ik->ij',
                emd.sigma_o_i[t],
                emd.theta_f[t] - emd.theta_o[t]
            )
            ddlpo = G + emd.sigma_o_i[t]
            ddlpo_i = np.linalg.inv(ddlpo)

            # Update theta_f
            emd.theta_f[t] -= np.einsum('ijk,ik->ij', ddlpo_i, dlpo)

            max_dlpo = np.amax(np.absolute(dlpo)) / emd.R
            iterations += 1

            if iterations == MAX_GA_ITERATIONS:
                raise Exception(
                    "The gradient-ascent algorithm did not converge before "
                    "reaching the maximum number of iterations."
                )

            emd.sigma_f[t] = ddlpo_i
            emd.sigma_f_i[t] = ddlpo

    return emd.theta_f, emd.sigma_f, emd.sigma_f_i, emd.sigma_o, emd.sigma_o_i

@njit
def cal_eta_G_Para(F1, theta):
    """
    Computes the first derivative (eta) and Fisher matrix (G) in a JIT-compiled manner for a single neuron.

    :param numpy.ndarray F1:
        Feature matrix of shape (R, n_params).
    :param numpy.ndarray theta:
        Parameter vector of shape (n_params,).

    :returns: tuple
        (eta, G)
        eta: numpy.ndarray of shape (n_params,)
        G: numpy.ndarray of shape (n_params, n_params)
    """
    R = F1.shape[0]
    temp = np.dot(F1, theta)
    r = np.empty(R)
    for i in range(R):
        r[i] = 1.0 / (1.0 + np.exp(-temp[i]))
    eta = np.dot(r, F1)
    p = r * (1.0 - r)
    G = np.dot(F1.T * p, F1)
    return eta, G

@njit
def process_single_i(theta_f_t_i, sigma_o_i_t_i, theta_o_t_i, FSUM_t_i,
                     F1, GA_CONVERGENCE, MAX_GA_ITERATIONS, R):
    """
    Performs Newton-Raphson updates for a single neuron's parameter vector using JIT compilation.

    :param numpy.ndarray theta_f_t_i:
        Current parameter vector of shape (n_params,).
    :param numpy.ndarray sigma_o_i_t_i:
        Inverse predicted covariance matrix of shape (n_params, n_params).
    :param numpy.ndarray theta_o_t_i:
        Predicted parameter vector (n_params,).
    :param numpy.ndarray FSUM_t_i:
        Feature sum array for the current neuron (n_params,).
    :param numpy.ndarray F1:
        Feature matrix (R, n_params).
    :param float GA_CONVERGENCE:
        Convergence threshold for gradient ascent updates.
    :param int MAX_GA_ITERATIONS:
        Maximum allowed iterations for gradient updates.
    :param int R:
        Number of trials.

    :returns: tuple
        (updated_theta, updated_inverse_hessian)
        updated_theta: numpy.ndarray of shape (n_params,)
        updated_inverse_hessian: numpy.ndarray of shape (n_params, n_params)
    """
    max_dlpo = np.inf
    iterations = 0
    while max_dlpo > GA_CONVERGENCE:
        eta, G = cal_eta_G_Para(F1, theta_f_t_i)
        dlpo = -(FSUM_t_i - eta) + np.dot(sigma_o_i_t_i, (theta_f_t_i - theta_o_t_i))
        ddlpo = G + sigma_o_i_t_i

        L = np.linalg.cholesky(ddlpo)
        y = np.linalg.solve(L, dlpo)
        x = np.linalg.solve(L.T, y)
        theta_f_t_i -= x

        max_dlpo = np.amax(np.absolute(dlpo)) / R
        iterations += 1
        if iterations == MAX_GA_ITERATIONS:
            raise Exception(
                "Gradient-ascent algorithm did not converge within "
                "MAX_GA_ITERATIONS."
            )

    # After convergence, compute the inverse Hessian via Cholesky factor
    I = np.eye(L.shape[0])
    L_inv = np.linalg.solve(L, I)
    ddlpo_i = np.dot(L_inv.T, L_inv)  # (L^-1).T * L^-1

    return theta_f_t_i, ddlpo_i

def filter_function_Parallel(emd):
    """
    Performs forward filtering on each neuron in parallel using joblib.

    :param emd: container.EMData
        The data structure with model parameters, spike data, etc.
    :returns: tuple
        Updated (theta_f, sigma_f, sigma_f_i, sigma_o, sigma_o_i).
    """
    GA_CONVERGENCE = 1e-5
    MAX_GA_ITERATIONS = 1000

    T = emd.T
    N = emd.N
    R = emd.R

    # Parameter dimension
    n_params = emd.theta_f[0, 0].shape[0]

    # Pre-allocate the feature matrix F1 for a single time step
    F1 = np.empty((R, n_params), dtype=emd.theta_f.dtype)
    F1[:, 0] = 1.0  # constant bias

    for t in range(T):
        if t == 0:
            emd.theta_o[0] = emd.init_theta
            emd.sigma_o[0] = emd.init_cov
            emd.sigma_o_i[0] = np.linalg.inv(emd.sigma_o[0])
        else:
            emd.theta_o[t] = emd.theta_f[t - 1]
            emd.sigma_o[t] = emd.sigma_f[t - 1] + emd.state_cov
            emd.sigma_o_i[t] = np.linalg.inv(emd.sigma_o[t])

        # Construct F1 by adding spikes data
        F1[:, 1:] = emd.spikes[t, :R]

        # Parallel processing over neurons
        results = Parallel(n_jobs=-1)(
            delayed(process_single_i)(
                emd.theta_f[t, i],
                emd.sigma_o_i[t, i],
                emd.theta_o[t, i],
                emd.FSUM[t, i],
                F1,
                GA_CONVERGENCE,
                MAX_GA_ITERATIONS,
                R
            )
            for i in range(N)
        )

        # Store results
        for i, (theta_f_i, ddlpo_i) in enumerate(results):
            emd.theta_f[t, i] = theta_f_i
            emd.sigma_f[t, i] = ddlpo_i
            emd.sigma_f_i[t, i] = np.linalg.inv(ddlpo_i)

    return emd.theta_f, emd.sigma_f, emd.sigma_f_i, emd.sigma_o, emd.sigma_o_i

def smoothing_function(emd):
    """
    Performs backward smoothing on the filtered estimates (non-parallel version).
    Complexity: O(T * N)

    :param emd: container.EMData
        Data structure containing filtered parameters, covariances, etc.
    :returns: tuple
        (sigma_s, theta_o, theta_s, lag_one_covariance, A)
        representing the smoothed covariances, predicted parameters, 
        smoothed parameters, lag-one covariance, and transition matrix A.
    """
    emd.theta_s[emd.T - 1] = emd.theta_f[emd.T - 1]
    emd.sigma_s[emd.T - 1] = emd.sigma_f[emd.T - 1]
    sigma_o_inv = np.linalg.inv(emd.sigma_o)

    for tt in range(emd.T - 1):
        t = emd.T - 2 - tt
        emd.A[t] = np.einsum('ijk,ikl->ijl', emd.sigma_f[t], sigma_o_inv[t + 1])

        tmp = np.einsum(
            'ijk,ik->ij',
            emd.A[t],
            (emd.theta_s[t + 1] - emd.theta_o[t + 1])
        )
        emd.theta_s[t] = emd.theta_f[t] + tmp

        tmp = np.einsum(
            'ijk,ikl->ijl',
            emd.A[t],
            (emd.sigma_s[t + 1] - emd.sigma_o[t + 1])
        )
        tmp = np.einsum(
            'ijk,ikl->ijl',
            tmp,
            emd.A[t].swapaxes(1, 2)
        )
        emd.sigma_s[t] = emd.sigma_f[t] + tmp
        emd.lag_one_covariance[t] = np.einsum(
            'ijk,ikl->ijl',
            emd.A[t],
            emd.sigma_s[t + 1]
        )

    return emd.sigma_s, emd.theta_o, emd.theta_s, emd.lag_one_covariance, emd.A

def process_single_i_smoothing(sigma_f_t_i, sigma_o_i_t1_i, theta_s_t1_i,
                               theta_o_t1_i, theta_f_t_i, sigma_s_t1_i, A_t_i):
    """
    Performs smoothing updates for a single neuron in the backward pass.

    :param numpy.ndarray sigma_f_t_i:
        Filtered covariance at time t for neuron i, shape (N+1, N+1).
    :param numpy.ndarray sigma_o_i_t1_i:
        Inverse of predicted covariance at time t+1 for neuron i, shape (N+1, N+1).
    :param numpy.ndarray theta_s_t1_i:
        Smoothed parameters at time t+1 for neuron i, shape (N+1,).
    :param numpy.ndarray theta_o_t1_i:
        Predicted parameters at time t+1 for neuron i, shape (N+1,).
    :param numpy.ndarray theta_f_t_i:
        Filtered parameters at time t for neuron i, shape (N+1,).
    :param numpy.ndarray sigma_s_t1_i:
        Smoothed covariance at time t+1 for neuron i, shape (N+1, N+1).
    :param numpy.ndarray A_t_i:
        Transition matrix from time t to t+1 for neuron i, shape (N+1, N+1).

    :returns: tuple
        (theta_s_t_i, sigma_s_t_i, A_t_i, lag_one_covariance_t_i)
        representing the updated smoothed parameters, smoothed covariance, 
        transition matrix, and lag-one covariance for neuron i at time t.
    """
    A_t_i = np.dot(sigma_f_t_i, sigma_o_i_t1_i)
    tmp_theta = np.dot(A_t_i, (theta_s_t1_i - theta_o_t1_i))
    theta_s_t_i = theta_f_t_i + tmp_theta

    tmp_sigma = np.dot(A_t_i, (sigma_s_t1_i - sigma_o_i_t1_i))
    tmp_sigma = np.dot(tmp_sigma, A_t_i.T)
    sigma_s_t_i = sigma_f_t_i + tmp_sigma

    lag_one_covariance_t_i = np.dot(A_t_i, sigma_s_t1_i)
    return theta_s_t_i, sigma_s_t_i, A_t_i, lag_one_covariance_t_i

def smoothing_function_Parallel(emd):
    """
    Performs backward smoothing on filtered estimates using parallel processing for each neuron.

    :param emd: container.EMData
        Data structure containing filtered parameters, covariances, etc.
    :returns: tuple
        (sigma_s, theta_o, theta_s, lag_one_covariance, A)
    """
    emd.theta_s[emd.T - 1] = emd.theta_f[emd.T - 1]
    emd.sigma_s[emd.T - 1] = emd.sigma_f[emd.T - 1]

    for tt in range(emd.T - 1):
        t = emd.T - 2 - tt

        results = Parallel(n_jobs=-1)(
            delayed(process_single_i_smoothing)(
                emd.sigma_f[t, i],
                emd.sigma_o_i[t + 1, i],
                emd.theta_s[t + 1, i],
                emd.theta_o[t + 1, i],
                emd.theta_f[t, i],
                emd.sigma_s[t + 1, i],
                emd.A[t, i]
            )
            for i in range(emd.N)
        )

        for i, (theta_s_t_i, sigma_s_t_i, A_t_i, lag_one_covariance_t_i) in enumerate(results):
            emd.theta_s[t, i] = theta_s_t_i
            emd.sigma_s[t, i] = sigma_s_t_i
            emd.A[t, i] = A_t_i
            emd.lag_one_covariance[t, i] = lag_one_covariance_t_i

    return emd.sigma_s, emd.theta_o, emd.theta_s, emd.lag_one_covariance, emd.A
