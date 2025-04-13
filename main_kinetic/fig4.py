"""
This code reproduces figures presented in:
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
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path, if needed
# (Adjust this depending on your project structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssll_kinetic
from ssll_kinetic import synthesis
import macro

import macro1
np.random.seed(1) 

def load_or_create_theta_spikes(
    N, T, R,
    coupling_mu, coupling_sigma, coupling_alpha,
    field_mu, field_sigma, field_alpha
):
    """
    Load or generate the pair (THETA, spikes) for the given parameters.
    If a file with the correct naming convention exists, it loads from it.
    Otherwise, it generates new data and saves it to file.

    Parameters
    ----------
    N : int
        Number of neurons.
    T : int
        Number of time bins.
    R : int
        Number of trials (spike simulations).
    coupling_mu : float
        Mean for the coupling parameters.
    coupling_sigma : float
        Standard deviation for the coupling parameters.
    coupling_alpha : float
        Correlation-scale for the coupling parameters.
    field_mu : float
        Mean for the field parameters.
    field_sigma : float
        Standard deviation for the field parameters.
    field_alpha : float
        Correlation-scale for the field parameters.

    Returns
    -------
    THETA : numpy.ndarray
        A 3D array of shape (T, N, N+1) containing the model parameters.
    spikes : numpy.ndarray
        A 3D array of shape (T+1, R, N) containing the simulated spike data.
    """
    directory = "saved_data/data_theta_spike"
    base_filename = (f"N{N}_T{T}_R{R}_"
                     f"cmu{coupling_mu}_csig{coupling_sigma}_calph{coupling_alpha}_"
                     f"fmu{field_mu}_fsig{field_sigma}_falph{field_alpha}")
    filename = f"{directory}/{base_filename}.pkl"

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        THETA, spikes = data['THETA'], data['spikes']
        print("THETA and spikes loaded from file.")
    else:
        print("No existing THETA/spikes file found; generating from scratch...")

        # Generate THETA for coupling
        THETA = ssll_kinetic.synthesis.get_THETA_gaussian_process(
            T, N, mu=coupling_mu, sigma=coupling_sigma, alpha=coupling_alpha
        )
        # Overwrite the field component
        THETA[:, :, 0] = ssll_kinetic.synthesis.generate_thetas(
            T, N, mu=field_mu, sigma=field_sigma, alpha=field_alpha
        )
        # Generate spikes
        spikes = ssll_kinetic.synthesis.get_S_function(T, R, N, THETA)

        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump({'THETA': THETA, 'spikes': spikes}, f)
        print("THETA and spikes have been generated and saved.")

    return THETA, spikes


def load_or_create_emd(
    N, T, R,
    coupling_mu, coupling_sigma, coupling_alpha,
    field_mu, field_sigma, field_alpha,
    spikes
):
    """
    Load the result of EM (EMD) from a pickle file if it exists.
    Otherwise, run EM to generate a new EMD and save it to file.

    Parameters
    ----------
    N : int
        Number of neurons.
    T : int
        Number of time bins.
    R : int
        Number of trials (spike simulations).
    coupling_mu : float
        Mean for the coupling parameters.
    coupling_sigma : float
        Standard deviation for the coupling parameters.
    coupling_alpha : float
        Correlation-scale for the coupling parameters.
    field_mu : float
        Mean for the field parameters.
    field_sigma : float
        Standard deviation for the field parameters.
    field_alpha : float
        Correlation-scale for the field parameters.
    spikes : numpy.ndarray
        Spike data of shape (T+1, R, N).

    Returns
    -------
    emd : object
        The EM fitting result (with attributes like emd.theta_s).
    """
    directory = "saved_data/data_emd"
    filename = (f"{directory}/emd_N{N}_T{T}_R{R}_"
                f"cmu{coupling_mu}_csig{coupling_sigma}_calph{coupling_alpha}_"
                f"fmu{field_mu}_fsig{field_sigma}_falph{field_alpha}.pkl")

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            emd = pickle.load(f)
        print("EMD loaded from file.")
    else:
        print("EMD file not found; running EM from scratch...")
        emd = ssll_kinetic.run(spikes, max_iter=120, mstep=True)
        with open(filename, 'wb') as f:
            pickle.dump(emd, f)
        print("EMD has been generated and saved.")
    return emd


def load_or_compute_sampling_entropy_flow(
    N, T, R_sampling,
    coupling_mu, coupling_sigma, coupling_alpha,
    field_mu, field_sigma, field_alpha,
    THETA,
    emd
):
    """
    Load or compute the sampling-based entropy flow for:
      - True parameters (THETA)
      - Estimated parameters (emd.theta_s)

    If a saved file exists, load the results. Otherwise, compute them from
    scratch, save them, and then return.

    Parameters
    ----------
    N : int
        Number of neurons.
    T : int
        Number of time bins.
    R_sampling : int
        Number of trials for spike sampling.
    coupling_mu : float
        Mean for the coupling parameters.
    coupling_sigma : float
        Standard deviation for the coupling parameters.
    coupling_alpha : float
        Correlation scale for the coupling parameters.
    field_mu : float
        Mean for the field parameters.
    field_sigma : float
        Standard deviation for the field parameters.
    field_alpha : float
        Correlation scale for the field parameters.
    THETA : numpy.ndarray
        True parameters of shape (T, N, N+1).
    emd : object
        Fitting result from EM, which provides emd.theta_s.

    Returns
    -------
    true_sampling_entropy_flow_total : numpy.ndarray
        The total sampling-based entropy flow (true parameters), shape (T,).
    true_sampling_entropy_flow_forward : numpy.ndarray
        The forward component (true parameters), shape (T,).
    true_sampling_entropy_flow_reverse : numpy.ndarray
        The reverse component (true parameters), shape (T,).
    est_sampling_entropy_flow_total : numpy.ndarray
        The total sampling-based entropy flow (estimated parameters), shape (T,).
    est_sampling_entropy_flow_forward : numpy.ndarray
        The forward component (estimated parameters), shape (T,).
    est_sampling_entropy_flow_reverse : numpy.ndarray
        The reverse component (estimated parameters), shape (T,).
    """
    directory = "saved_data/data_sampling_entropy_flow"
    filename = (f"{directory}/sampling_entropy_flow_N{N}_T{T}_R{R_sampling}_"
                f"mu{coupling_mu}_sigma{coupling_sigma}_alpha{coupling_alpha}_"
                f"fieldmu{field_mu}_fieldsigma{field_sigma}_fieldalpha{field_alpha}.pkl")
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            (true_sampling_entropy_flow_total,
             true_sampling_entropy_flow_forward,
             true_sampling_entropy_flow_reverse,
             est_sampling_entropy_flow_total,
             est_sampling_entropy_flow_forward,
             est_sampling_entropy_flow_reverse) = pickle.load(f)
        print("Sampling-based entropy flow data loaded from file.")
    else:
        print("No sampling-based entropy flow file found; generating from scratch...")

        # 1) Sampling-based entropy flow for TRUE parameters
        (true_sampling_entropy_flow_total,
         true_sampling_entropy_flow_forward,
         true_sampling_entropy_flow_reverse) = simulate_spikes_and_sampling_entropy_flow(
             T, R_sampling, N, THETA
         )

        # 2) Sampling-based entropy flow for ESTIMATED parameters
        (est_sampling_entropy_flow_total,
         est_sampling_entropy_flow_forward,
         est_sampling_entropy_flow_reverse) = simulate_spikes_and_sampling_entropy_flow(
             T, R_sampling, N, emd.theta_s
         )

        # Save all arrays
        with open(filename, 'wb') as f:
            pickle.dump((true_sampling_entropy_flow_total,
                         true_sampling_entropy_flow_forward,
                         true_sampling_entropy_flow_reverse,
                         est_sampling_entropy_flow_total,
                         est_sampling_entropy_flow_forward,
                         est_sampling_entropy_flow_reverse), f)

        print("Sampling-based entropy flow data has been generated and saved.")

    return (true_sampling_entropy_flow_total,
            true_sampling_entropy_flow_forward,
            true_sampling_entropy_flow_reverse,
            est_sampling_entropy_flow_total,
            est_sampling_entropy_flow_forward,
            est_sampling_entropy_flow_reverse)


def simulate_spikes_and_sampling_entropy_flow(T, R_sampling, N, THETA):
    """
    Simulate spikes via sampling using the given kinetic Ising model parameters (THETA),
    then compute the total, forward, and reverse sampling-based entropy flow.

    Parameters
    ----------
    T : int
        Number of time bins.
    R_sampling : int
        Number of trials for spike sampling.
    N : int
        Number of neurons.
    THETA : numpy.ndarray
        Model parameters of shape (T, N, N+1).

    Returns
    -------
    sampling_entropy_flow_total : numpy.ndarray
        Total sampling-based entropy flow (forward - reverse) at each time step,
        shape (T,).
    sampling_entropy_flow_forward : numpy.ndarray
        Forward component of the sampling-based entropy flow, shape (T,).
    sampling_entropy_flow_reverse : numpy.ndarray
        Reverse component of the sampling-based entropy flow, shape (T,).
    """
    # (Same internal computation as before; only variable names changed.)

    # Allocate arrays for intermediate sampling-based calculations
    sampling_psi = np.zeros((T, R_sampling, N))
    sampling_spikes = np.zeros((T + 1, R_sampling, N))
    sampling_rand_numbers = np.random.rand(T + 1, R_sampling, N)

    # Initialize spike states randomly
    sampling_spikes[0] = (sampling_rand_numbers[0] >= 0.5).astype(int)

    # Generate spike data by sampling
    for t in range(1, T + 1):
        # Concatenate a column of 1's for the bias term
        sampling_F_psi = np.concatenate([
            np.ones((R_sampling, 1)),
            sampling_spikes[t - 1]
        ], axis=1)

        # Calculate log-partition (psi) for each sample and neuron
        sampling_psi[t - 1] = np.log(1 + np.exp(THETA[t - 1] @ sampling_F_psi.T)).T

        # Probability of spiking
        sampling_p_spike = np.exp(THETA[t - 1] @ sampling_F_psi.T 
                                  - sampling_psi[t - 1].T).T

        # Draw spikes based on sampling_p_spike
        sampling_spikes[t] = (sampling_p_spike >= sampling_rand_numbers[t]).astype(int)

    # Prepare arrays for the forward direction
    FSUM_sampling_forward = np.zeros((T, R_sampling, N, N + 1))
    FSUM_sampling_forward_one = np.zeros((T, R_sampling, N, N + 1))

    # Build design matrices for the forward part
    for l in range(R_sampling):
        for tt in range(1, T + 1):
            for n in range(N):
                FSUM_sampling_forward[tt - 1, l, n] = np.append(
                    sampling_spikes[tt, l, n],
                    sampling_spikes[tt, l, n] * sampling_spikes[tt - 1, l]
                )
                FSUM_sampling_forward_one[tt - 1, l, n] = np.append(
                    1,
                    sampling_spikes[tt - 1, l]
                )

    # Average over trials in the forward direction
    FSUM_sampling_forward_mean = FSUM_sampling_forward.mean(axis=1)

    # Compute the forward log-partition
    phi_sampling_forward = np.zeros((T, R_sampling, N))
    for l in range(R_sampling):
        for tt in range(T):
            for nn in range(N):
                phi_sampling_forward[tt, l, nn] = np.log(
                    1 + np.exp(
                        THETA[tt, nn] @ FSUM_sampling_forward_one[tt, l, nn]
                    )
                )
    phi_sampling_forward = phi_sampling_forward.mean(axis=1).sum(axis=1)

    # Prepare arrays for the reverse direction
    FSUM_sampling_reverse = np.zeros((T, R_sampling, N, N + 1))
    FSUM_sampling_reverse_one = np.zeros((T, R_sampling, N, N + 1))

    # Build design matrices for the reverse part
    for l in range(R_sampling):
        for tt in range(1, T + 1):
            for n in range(N):
                FSUM_sampling_reverse[tt - 1, l, n] = np.append(
                    sampling_spikes[tt - 1, l, n],
                    sampling_spikes[tt - 1, l, n] * sampling_spikes[tt, l]
                )
                FSUM_sampling_reverse_one[tt - 1, l, n] = np.append(
                    1,
                    sampling_spikes[tt, l]
                )

    # Average over trials in the reverse direction
    FSUM_sampling_reverse_mean = FSUM_sampling_reverse.mean(axis=1)

    # Compute the reverse log-partition
    phi_sampling_reverse = np.zeros((T, R_sampling, N))
    for l in range(R_sampling):
        for tt in range(T):
            for nn in range(N):
                phi_sampling_reverse[tt, l, nn] = np.log(
                    1 + np.exp(
                        THETA[tt, nn] @ FSUM_sampling_reverse_one[tt, l, nn]
                    )
                )
    phi_sampling_reverse = phi_sampling_reverse.mean(axis=1).sum(axis=1)

    # Forward and reverse sampling-based entropy flow components
    sampling_entropy_flow_forward = (THETA * FSUM_sampling_forward_mean).sum(axis=(1, 2)) \
                                    - phi_sampling_forward
    sampling_entropy_flow_reverse = (THETA * FSUM_sampling_reverse_mean).sum(axis=(1, 2)) \
                                    - phi_sampling_reverse

    # Total sampling-based entropy flow
    sampling_entropy_flow_total = sampling_entropy_flow_forward - sampling_entropy_flow_reverse

    return (sampling_entropy_flow_total,
            sampling_entropy_flow_forward,
            sampling_entropy_flow_reverse)


def compute_mean_field_entropy_flow(emd, param_array, macro):
    """
    Compute the total mean-field-based entropy flow (forward - reverse)
    for each time step, given EM fitting results, a parameter array
    (true or estimated), and the macro module that provides
    the necessary computation functions.
    ...
    """
    T = emd.T
    N = emd.N

    # This array will store the total entropy flow at each time step
    entropy_flow = np.zeros(T)

    # Initialize magnetization using the average of the spike data
    m_prev = np.mean(emd.spikes, axis=(0, 1))

    for t in range(T):
        if t == 0:
            m_p = m_prev
        else:
            m_p = m

        # Extract the parameter set for time t
        theta_t = param_array[t]

        # Compute the magnetization via mean-field
        m = macro.computation_m(theta_t, m_p)

        # Destructure the return from Dissipation_en with the new variable names
        (entropy_flow_forward_t,
         entropy_flow_reverse_t,
         entropy_flow_t,
         _,
         _,
         _) = macro.Dissipation_en(theta_t, m, m_p)

        # We only store the total entropy flow (forward - reverse)
        entropy_flow[t] = entropy_flow_t
    
    
    return entropy_flow


import matplotlib.pyplot as plt
import os

def plot_entropy_comparison(
    true_sampling_entropy_flow_total,
    est_entropy_flow,
    true_entropy_flow,
    est_sampling_entropy_flow_total,
    y_lim=None,
    linewidth=4,
    fontsize_label=20,
    fontsize_title=24,
    fontsize_legend=20,
    fontsize_ticklabel=16
):
    """
    Plot four curves comparing different entropy flow estimation methods:
    - Sampling with true parameters
    - Mean-field with estimated parameters
    - Mean-field with true parameters
    - Sampling with estimated parameters

    Parameters
    ----------
    true_sampling_entropy_flow_total : numpy.ndarray
        Entropy flow computed via sampling-based approach with true parameters; shape (T,).
    est_entropy_flow : numpy.ndarray
        Entropy flow computed via mean-field approach with estimated parameters; shape (T,).
    true_entropy_flow : numpy.ndarray
        Entropy flow computed via mean-field approach with true parameters; shape (T,).
    est_sampling_entropy_flow_total : numpy.ndarray
        Entropy flow computed via sampling-based approach with estimated parameters; shape (T,).
    y_lim : tuple, optional
        (y_min, y_max) for the y-axis limit.
    linewidth : float, optional
        Line width for the plots.
    fontsize_label : int, optional
        Font size for the x/y axis labels.
    fontsize_title : int, optional
        Font size for the plot title (unused if no title).
    fontsize_legend : int, optional
        Font size for the legend.
    fontsize_ticklabel : int, optional
        Font size for the tick labels.
    """

    plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlabel("Time [AU]", fontsize=fontsize_label)

    # 1) Sampling with true parameters
    ax1.plot(
        range(len(true_sampling_entropy_flow_total)),
        true_sampling_entropy_flow_total,
        label='Sampling (true params)',
        color='blue',
        linewidth=linewidth
    )

    # 2) Sampling with estimated parameters
    ax1.plot(
        range(len(est_sampling_entropy_flow_total)),
        est_sampling_entropy_flow_total,
        label='Sampling (estimated params)',
        color='green',
        linewidth=linewidth
    )

    # 3) Mean-field with true parameters
    ax1.plot(
        range(len(true_entropy_flow)),
        true_entropy_flow,
        label='Mean-field (true params)',
        color='black',
        linestyle='-',
        linewidth=linewidth
    )

    # 4) Mean-field with estimated parameters
    ax1.plot(
        range(len(est_entropy_flow)),
        est_entropy_flow,
        label='Mean-field (estimated params)',
        color='red',
        linewidth=linewidth
    )

    # Adjust x-axis limit
    ax1.set_xlim(3, len(true_sampling_entropy_flow_total))

    # Adjust y-axis limit if provided
    if y_lim is not None:
        ax1.set_ylim(y_lim)
    else:
        ax1.set_ylim(bottom=0)

    ax1.legend(fontsize=fontsize_legend)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticklabel)
    plt.tight_layout()

    # Create 'fig/' directory if it doesn't exist
    if not os.path.exists("fig"):
        os.mkdir("fig")

    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Save without a plot title; filenames are changed to the requested name
    plt.savefig('fig/figure4.pdf', format='pdf')
    plt.savefig('fig/figure14.eps', format='eps')
    plt.show()


if __name__ == "__main__":
    """
    Main execution part: toggles for loading/saving or generating data from scratch,
    EM fitting, sampling-based entropy calculation, and final plotting.
    """

    # ---------------------------
    # 1) Set up parameters
    # ---------------------------
    N = 80
    T = 150
    R = 500
    R_sampling = 500

    coupling_mu = 5 / N
    coupling_sigma = 30
    coupling_alpha = 0.1 * N

    field_mu = -3
    field_sigma = 50
    field_alpha = 1

    # ---------------------------
    # 2) Load or generate the true parameters and spikes
    # ---------------------------
    USE_SAVED_THETA = True
    if USE_SAVED_THETA:
        THETA, spikes = load_or_create_theta_spikes(
            N, T, R,
            coupling_mu, coupling_sigma, coupling_alpha,
            field_mu, field_sigma, field_alpha
        )
    else:
        THETA = ssll_kinetic.synthesis.get_THETA_gaussian_process(
            T, N, mu=coupling_mu, sigma=coupling_sigma, alpha=coupling_alpha
        )
        THETA[:, :, 0] = ssll_kinetic.synthesis.generate_thetas(
            T, N, mu=field_mu, sigma=field_sigma, alpha=field_alpha
        )
        spikes = ssll_kinetic.synthesis.get_S_function(T, R, N, THETA)

    # ---------------------------
    # 3) Load or run EMD
    # ---------------------------
    USE_SAVED_EMD = False#True
    if USE_SAVED_EMD:
        emd = load_or_create_emd(
            N, T, R,
            coupling_mu, coupling_sigma, coupling_alpha,
            field_mu, field_sigma, field_alpha,
            spikes
        )
    else:
        emd = ssll_kinetic.run(spikes, max_iter=120, mstep=True)

    # ---------------------------
    # 4) Load or compute sampling-based Entropy
   # ------------------------------------------------
    # Example usage in your main script
    # ------------------------------------------------
    USE_SAVED_SAMPLING = True
    if USE_SAVED_SAMPLING:
        (true_sampling_entropy_flow_total,
        true_sampling_entropy_flow_forward,
        true_sampling_entropy_flow_reverse,
        est_sampling_entropy_flow_total,
        est_sampling_entropy_flow_forward,
        est_sampling_entropy_flow_reverse) = load_or_compute_sampling_entropy_flow(
            N, T, R_sampling,
            coupling_mu, coupling_sigma, coupling_alpha,
            field_mu, field_sigma, field_alpha,
            THETA, emd
        )
    else:
        # If not using saved data, compute directly from scratch
        (true_sampling_entropy_flow_total,
        true_sampling_entropy_flow_forward,
        true_sampling_entropy_flow_reverse) = \
            simulate_spikes_and_sampling_entropy_flow(T, R_sampling, N, THETA)

        (est_sampling_entropy_flow_total,
        est_sampling_entropy_flow_forward,
        est_sampling_entropy_flow_reverse) = \
            simulate_spikes_and_sampling_entropy_flow(T, R_sampling, N, emd.theta_s)
    # ---------------------------
    # 5) Mean-field from true / estimated parameters
    # ---------------------------
    # Compute mean-field bath entropy for true parameters
    true_entropy_flow = compute_mean_field_entropy_flow(emd, THETA, macro)
    # Compute mean-field entropy flow for estimated parameters
    est_entropy_flow = compute_mean_field_entropy_flow(emd, emd.theta_s, macro)   
    # ---------------------------
    # 6) Plot a comparison
    # ---------------------------
    plot_entropy_comparison(
        true_sampling_entropy_flow_total,
        est_entropy_flow,
        true_entropy_flow,
        est_sampling_entropy_flow_total,
        y_lim=(0, 25)
    )

   
