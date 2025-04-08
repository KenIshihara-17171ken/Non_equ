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
import pdb
from ssll_kinetic.probability import *
import time

# Set function pointer for log marginal likelihood computation.
log_marginal_functions = log_marginal

class EMData:
    def __init__(self, spikes):
        """
        Container class for storing and managing the results of the EM algorithm
        for the state-space kinetic Ising model. This class holds the spike data,
        parameters, covariances, and timing information required for filtering,
        smoothing, and parameter estimation over time.

        The complexity of the initialization is O(1) with respect to the model fitting,
        although loops are used to set up arrays for each neuron.

        :param numpy.ndarray spikes:
            A binary spike array with dimensions (T+1, R, N) where:
              - T+1: number of time bins (including an initial time t=0),
              - R: number of trials (or runs),
              - N: number of neurons.
              A value of 1 indicates a spike and 0 indicates no spike.

        Attributes:
            spikes (numpy.ndarray): Input spike data.
            T (int): Number of time bins used in the model (spikes.shape[0] - 1).
            R (int): Number of trials (spikes.shape[1]).
            N (int): Number of neurons (spikes.shape[2]).
            dim_pram (int): Dimension of the parameter vector (initialized to 0).
            aic (float): Akaike Information Criterion (initialized to 0).
            
            state_cov (numpy.ndarray): An array of shape (N, N+1, N+1) containing the 
                state covariance matrices for each neuron. Initialized to 0.5 * identity.
            F (numpy.ndarray): An array of shape (N, N+1, N+1) representing the state 
                transition matrices for each neuron. Initialized to identity.
            init_cov (numpy.ndarray): Initial covariance matrices for each neuron 
                (shape: (N, N+1, N+1)), initialized to identity.
            init_cov_i (numpy.ndarray): Inverse of the initial covariance matrices.
            init_theta (numpy.ndarray): Initial parameter (theta) estimates for each neuron 
                (shape: (N, N+1)), initialized to zeros.
            
            e_step_time, filtering_time, smoothing_time, m_step_time, llk_time (float):
                Timing metrics for the E-step, filtering, smoothing, M-step, and log likelihood
                computation.
            FSUM (numpy.ndarray): Pre-computed array of shape (T, N, N+1) used for accumulating
                spike-based sums. Computed over trials and time.
            
            theta_o (numpy.ndarray): "Observation" theta parameters of shape (T, N, N+1).
            theta_f (numpy.ndarray): Filtered theta estimates (shape: (T, N, N+1)).
            theta_s (numpy.ndarray): Smoothed theta estimates (shape: (T, N, N+1)).
            eta_s (numpy.ndarray): Smoothed expectation parameters (shape: (T, N, N+1)).
            
            sigma_o (numpy.ndarray): Observation covariance matrices 
                (shape: (T, N, N+1, N+1)).
            sigma_f (numpy.ndarray): Filtered covariance matrices 
                (shape: (T, N, N+1, N+1)).
            sigma_f_i (numpy.ndarray): Inverse of filtered covariance matrices.
            sigma_o_i (numpy.ndarray): Inverse of observation covariance matrices.
            sigma_s (numpy.ndarray): Smoothed covariance matrices (shape: (T, N, N+1, N+1)),
                initialized to ones.
            
            A (numpy.ndarray): Lag-one state transition matrices (shape: (T-1, N, N+1, N+1)).
            lag_one_covariance (numpy.ndarray): Lag-one covariance matrices 
                (shape: (T, N, N+1, N+1)).
            
            marg_llk (function): Function pointer to compute the marginal log likelihood.
            mllk (float): Current marginal log likelihood (initialized to infinity).
            mllk_list (list): List to store the marginal log likelihood values over iterations.
            iterations_list (list): List to record iteration numbers.
            Q_list (list): List to store Q matrices (state covariance) over iterations.
            F_list (list): List to store F matrices over iterations.
            iterations (int): Counter for the number of EM iterations.
            CONVERGED (float): Convergence threshold (default: 1e-5).
            convergence (float): Current convergence value (initialized to infinity).
        """
        self.spikes = spikes
        self.T = spikes.shape[0] - 1
        self.R = spikes.shape[1]
        self.N = spikes.shape[2]

        self.dim_pram = 0
        self.aic = 0

        # Initialize state covariance matrices for each neuron as 0.5 * identity.
        self.state_cov = np.zeros((self.N, self.N+1, self.N+1))
        for i in range(self.N):
            self.state_cov[i] = 0.5 * np.identity(self.N+1)

        # Initialize F matrices as identity for each neuron.
        self.F = np.zeros((self.N, self.N+1, self.N+1))
        for i in range(self.N):
            self.F[i] = 1 * np.identity(self.N+1)

        # Initialize initial covariance matrices as identity.
        self.init_cov = np.zeros((self.N, self.N+1, self.N+1))
        for i in range(self.N):
            self.init_cov[i] = np.identity(self.N+1)
        self.init_cov_i = np.linalg.inv(self.init_cov)
        
        # Initialize initial theta parameters as zeros.
        self.init_theta = 0 * np.ones((self.N, self.N+1))
        
        # Timing metrics.
        self.e_step_time = 0
        self.filtering_time = 0
        self.smoothing_time = 0
        self.m_step_time = 0
        self.llk_time = 0
        
        # Pre-compute FSUM: an array that sums spike activity and their products across trials.
        self.FSUM = np.zeros((self.T, self.N, self.N+1))
        for l in range(self.R):
            for t in range(1, self.T+1):
                for n in range(self.N):
                    # Append spike value and product with previous time step's spike.
                    self.FSUM[t-1, n] += np.append(self.spikes[t, l, n],
                                                     self.spikes[t, l, n] * self.spikes[t-1, l])
        # Initialize theta estimates.
        self.theta_o = np.zeros((self.T, self.N, self.N+1))
        self.theta_f = np.zeros((self.T, self.N, self.N+1))
        self.theta_s = np.zeros((self.T, self.N, self.N+1))
        self.eta_s = np.zeros((self.T, self.N, self.N+1))  # Smoothed expectations.
        
        # Initialize covariance estimates.
        self.sigma_o = np.zeros((self.T, self.N, self.N+1, self.N+1))
        self.sigma_f = np.zeros((self.T, self.N, self.N+1, self.N+1))
        for t in range(self.T):
            for i in range(self.N):
                self.sigma_f[t, i] = np.identity(self.N+1)
                self.sigma_o[t, i] = np.identity(self.N+1)
        self.sigma_f_i = np.linalg.inv(self.sigma_f)
        self.sigma_o_i = np.linalg.inv(self.sigma_o)
        self.sigma_s = np.ones((self.T, self.N, self.N+1, self.N+1))
        
        # Initialize transition and lag-one covariance matrices.
        self.A = np.zeros((self.T-1, self.N, self.N+1, self.N+1))
        self.lag_one_covariance = np.zeros((self.T, self.N, self.N+1, self.N+1))
        
        # Set the marginal log likelihood function.
        self.marg_llk = log_marginal_functions        
        self.mllk = np.inf
        self.mllk_list = []
        self.iterations_list = []
        self.Q_list = []
        self.F_list = []
        self.iterations = 0
        self.CONVERGED = 1e-5
        self.convergence = np.inf
