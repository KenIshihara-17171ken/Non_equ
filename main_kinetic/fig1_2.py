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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

# If "ssll_kinetic" is in a parent folder, uncomment the following line:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssll_kinetic
from ssll_kinetic import synthesis

# ===================================================================
# 1) Generate Theta
# ===================================================================
def generate_theta(T, N):
    """
    Generates the time-varying parameter array Theta for N neurons over T time bins.
    The shape of Theta is (T, N, N+1):
      - Theta[:, :, 0] = "field" parameters (one per neuron)
      - Theta[:, :, 1..N] = "coupling" parameters between neurons

    :param int T:
        Number of time bins.
    :param int N:
        Number of neurons.

    :returns:
        numpy.ndarray
            A 3D array of shape (T, N, N+1) containing the generated parameters.
    """
    # Mean and variance for coupling
    coupling_mu = 5 / N
    coupling_sigma = 30
    coupling_alpha = 0.1 * N

    # Mean and variance for field
    field_mu = -3
    field_sigma = 50
    field_alpha = 1

    # Generate coupling parameters (Theta[:, :, 1..N])
    THETA = synthesis.get_THETA_gaussian_process(
        T, N, mu=coupling_mu, sigma=coupling_sigma, alpha=coupling_alpha
    )
    # Generate field parameters (Theta[:, :, 0])
    tmp = synthesis.generate_thetas(
        T, N, mu=field_mu, sigma=field_sigma, alpha=field_alpha
    )
    THETA[:, :, 0] = tmp
    return THETA

# ===================================================================
# 2) Generate Spikes
# ===================================================================
def generate_spikes(T, R, N, THETA):
    """
    Generates spike data based on the state-space kinetic Ising model.

    :param int T:
        Number of time bins.
    :param int R:
        Number of trials (or runs).
    :param int N:
        Number of neurons.
    :param numpy.ndarray THETA:
        The parameter array of shape (T, N, N+1) used to generate spikes.

    :returns:
        numpy.ndarray
            A 3D binary array of shape (T, R, N), where a 1 indicates a spike.
    """
    return synthesis.get_S_function(T, R, N, THETA)

# ===================================================================
# 3) Plotting Utilities for N=2
# ===================================================================
fontsize_title = 20
fontsize_label = 16
fontsize_legend = 14
fontsize_ABC = 20
fontsize_ticklabel = 14
fontsize_tick = 6

def plot_theta_image(ax, title, scale=1.5):
    """
    Loads and plots an illustrative image for the 2-neuron case.

    :param matplotlib.axes.Axes ax:
        The Axes object on which to draw the image.
    :param str title:
        The title of the subplot.
    :param float scale:
        Scaling factor for the displayed image size.

    :returns:
        None
    """
    im = Image.open("neuron2.jpg")
    im_list = np.asarray(im)
    orig_size = im_list.shape[1], im_list.shape[0]
    extent = [0, orig_size[0] * scale, 0, orig_size[1] * scale]
    ax.imshow(im_list, extent=extent, aspect='equal')
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize_title)

def plot_spikes(ax, spikes, neuron_idx, title):
    """
    Plots the spike raster for a single neuron across time bins (x-axis) and trials (y-axis).

    :param matplotlib.axes.Axes ax:
        The Axes object on which to draw the spike raster.
    :param numpy.ndarray spikes:
        A 3D binary array of shape (T, R, N).
    :param int neuron_idx:
        The index of the neuron to be plotted.
    :param str title:
        Title for the subplot.

    :returns:
        None
    """
    ax.imshow(spikes[:, :, neuron_idx].T, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel('Time [AU]', fontsize=fontsize_label)
    ax.tick_params(axis='both', labelsize=fontsize_label, length=fontsize_tick)
    if neuron_idx == 0:
        ax.set_ylabel('Trial', fontsize=fontsize_label)

def plot_mllk(ax, emd, title):
    """
    Plots the log marginal likelihood across EM iterations.

    :param matplotlib.axes.Axes ax:
        The Axes object on which to draw the plot.
    :param container.EMData emd:
        The fitted model data, which contains the mllk_list attribute.
    :param str title:
        Title for the subplot.

    :returns:
        None
    """
    ax.plot(emd.mllk_list)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlim([0, len(emd.mllk_list)])
    ax.tick_params(axis='both', labelsize=fontsize_ticklabel, length=fontsize_tick)
    ax.set_xlabel('Number of EM iterations', fontsize=fontsize_label)

def plot_Q(ax, Q, neuron_idx, fig, vmin=None, vmax=None, tick_labelsize=10):
    """
    Plots the hyper-parameter Q for a given neuron.

    :param matplotlib.axes.Axes ax:
        The Axes object on which to draw the heatmap.
    :param list Q:
        A list of Q matrices (one per neuron). Typically, Q_list[-1][neuron_idx] is the final matrix.
    :param int neuron_idx:
        Index of the neuron whose Q matrix will be plotted.
    :param matplotlib.figure.Figure fig:
        The Figure object (for colorbar management).
    :param float vmin:
        Minimum value for color scaling.
    :param float vmax:
        Maximum value for color scaling.
    :param int tick_labelsize:
        Font size for colorbar tick labels.

    :returns:
        matplotlib.colorbar.Colorbar
            The colorbar associated with the Q matrix heatmap.
    """
    mat = Q[neuron_idx]
    aximg = ax.imshow(
        mat, interpolation='nearest', cmap='RdGy',
        aspect='equal', alpha=1, vmin=vmin, vmax=vmax
    )
    ax.axis('off')
    ax.set_title(r'$Q_{%d}$' % (neuron_idx + 1), fontsize=fontsize_title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(aximg, cax=cax)
    for label in cbar.ax.get_yticklabels():
        label.set_size(tick_labelsize)
    return cbar

def plot_theta_estimates(gs, emd, THETA, N, T):
    """
    Plots the estimated parameters vs. the true parameters for N neurons:
     - Field parameters (Theta[:, :, 0])
     - Coupling parameters (Theta[:, :, 1..N])

    :param matplotlib.gridspec.GridSpec gs:
        GridSpec object for arranging subplots.
    :param container.EMData emd:
        The fitted model data, containing posterior means and covariances of Theta.
    :param numpy.ndarray THETA:
        The true parameter array of shape (T, N, N+1).
    :param int N:
        Number of neurons.
    :param int T:
        Number of time bins.

    :returns:
        None
    """
    # Field parameter
    ax1 = plt.subplot(gs[0, 1])
    for i in range(N):
        label_est = r'$\theta_{%d}$ estimated' % (i + 1)
        label_true = r'$\theta_{%d}$ true' % (i + 1)
        ax1.plot(emd.theta_s[:, i, 0], label=label_est)
        ax1.plot(THETA[:, i, 0], color='black', linestyle='--', label=label_true)
        ax1.legend(fontsize=fontsize_legend, loc='lower right')
        x_min = emd.theta_s[:, i, 0] - 1.645 * np.sqrt(emd.sigma_s[:, i, 0, 0])
        x_max = emd.theta_s[:, i, 0] + 1.645 * np.sqrt(emd.sigma_s[:, i, 0, 0])
        ax1.fill_between(range(T), x_min, x_max, alpha=0.2)
    ax1.set_title(r'Field parameters, $\theta_{i,t}$', fontsize=fontsize_title)
    ax1.set_xlim([0, T])
    ax1.tick_params(axis='both', labelsize=fontsize_ticklabel, length=fontsize_tick)

    # Coupling parameters
    for i in range(N):
        ax2 = plt.subplot(gs[i + 1, 1])
        for j in range(N):
            ix = j + 1
            confbound = 1.96
            x_min = emd.theta_s[:, i, ix] - confbound * np.sqrt(emd.sigma_s[:, i, ix, ix])
            x_max = emd.theta_s[:, i, ix] + confbound * np.sqrt(emd.sigma_s[:, i, ix, ix])
            label_est = r'$\theta_{{%d,%d}}$ estimated' % (i + 1, j + 1)
            label_true = r'$\theta_{{%d,%d}}$ true' % (i + 1, j + 1)
            ax2.plot(emd.theta_s[:, i, ix], label=label_est)
            ax2.plot(THETA[:, i, ix], color='black', linestyle='--', label=label_true)
            ax2.fill_between(range(T), x_min, x_max, alpha=0.5)
        ax2.legend(fontsize=fontsize_legend, loc='lower right')
        ax2.set_title(r'Coupling parameters, $\theta_{%d j,t}$' % (i + 1), fontsize=fontsize_title)
        ax2.set_xlim([0, T])
        ax2.tick_params(axis='both', labelsize=fontsize_ticklabel, length=fontsize_tick)
    ax2.set_xlabel('Time [AU]', fontsize=fontsize_label)

def plot_results_n2(emd, THETA, spikes, N, T):
    """
    Plots the main results for the case N=2:
      - Neuron illustration
      - Spike rasters for each neuron
      - Log marginal likelihood over EM iterations
      - Hyper-parameter Q
      - Estimated vs. true parameters

    :param container.EMData emd:
        The fitted model data for N=2.
    :param numpy.ndarray THETA:
        The true parameter array of shape (T, 2, 3).
    :param numpy.ndarray spikes:
        A 3D binary array of shape (T, R, 2).
    :param int N:
        Number of neurons (2 in this case).
    :param int T:
        Number of time bins.

    :returns:
        None
    """
    fig = plt.figure(figsize=(30, 25))
    gs1 = gridspec.GridSpec(3, 8)
    gs2 = gridspec.GridSpec(3, 2)

    # A) Image of 2 neurons
    ax = plt.subplot(gs1[0, 0:4])
    plot_theta_image(ax, "2 neurons")

    # B) Spikes of neuron1 and neuron2
    ax1 = plt.subplot(gs1[1, 0:2])
    plot_spikes(ax1, spikes, 0, 'Neuron 1')
    ax2 = plt.subplot(gs1[1, 2:4])
    plot_spikes(ax2, spikes, 1, 'Neuron 2')
    plt.yticks([])

    # C) Log marginal likelihood
    ax = plt.subplot(gs1[2, 0:2])
    plot_mllk(ax, emd, "Log marginal likelihood")

    # D, E) Hyper-parameter Q
    vmin = np.percentile([q for q in emd.Q_list], 10)
    vmax = np.percentile([q for q in emd.Q_list], 90)
    first_cbar = None
    for i in range(N):
        ax = plt.subplot(gs1[2, i + 2])
        cbar = plot_Q(ax, emd.Q_list[-1], i, fig, vmin, vmax, tick_labelsize=(fontsize_ticklabel - 5))
        if i == 0:
            first_cbar = cbar
    # Remove the first colorbar to avoid duplication
    if first_cbar:
        first_cbar.remove()

    # Parameter estimates
    plot_theta_estimates(gs2, emd, THETA, N, T)

    # Optional labeling
    labels = ['A', 'B', 'C', 'D', 'E']
    label_x_positions = [0.11, 0.11, 0.11, 0.32, 0.52]
    label_y_positions = [0.9, 0.63, 0.32, 0.3, 0.9]
    for x, y, label in zip(label_x_positions, label_y_positions, labels):
        ax = fig.add_axes([x, y, .05, .05], frameon=False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.0, 0.0, label, fontsize=fontsize_ABC, fontweight='bold')

    # Title for Q
    ax = fig.add_axes([0.35, 0.3, .05, .05], frameon=False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(0.0, 0.0, r'Hyper-parameter $Q_{i}$', fontsize=fontsize_title)
    
    if not os.path.exists("fig"):
        os.mkdir("fig")
    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.savefig('fig/figure1.eps', bbox_inches='tight')
    plt.savefig('fig/figure1.pdf', bbox_inches='tight')
    plt.show()

# ===================================================================
# 4) Plotting Utilities for N=12
# ===================================================================
def display_neuron_image_and_spikes(spikes, gs):
    """
    Plots spike rasters for 3 example trials: 1, 100, and 200.
    Assumes R >= 200 trials.

    :param numpy.ndarray spikes:
        A 3D binary array of shape (T, R, N).
    :param matplotlib.gridspec.GridSpec gs:
        A GridSpec object for arranging subplots.

    :returns:
        None
    """
    ax1 = plt.subplot(gs[0, 0:2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:6])

    ax1.imshow(spikes[:, 0, :].T, cmap='binary', aspect='auto', interpolation='nearest')
    ax1.set_yticks([])
    ax1.set_title('Trial 1', fontsize=fontsize_title)
    ax1.set_xlabel('Time [AU]', fontsize=fontsize_label)
    ax1.set_ylabel('Neuron', fontsize=fontsize_label)

    ax2.imshow(spikes[:, 100, :].T, cmap='binary', aspect='auto', interpolation='nearest')
    ax2.set_yticks([])
    ax2.set_title('Trial 100', fontsize=fontsize_title)
    ax2.set_xlabel('Time [AU]', fontsize=fontsize_label)

    ax3.imshow(spikes[:, -1, :].T, cmap='binary', aspect='auto', interpolation='nearest')
    ax3.set_yticks([])
    ax3.set_title('Trial 200', fontsize=fontsize_title)
    ax3.set_xlabel('Time [AU]', fontsize=fontsize_label)

def display_selected_neuron_theta(fig, gs, emd, THETA, selected_neurons):
    """
    Plots time-varying coupling parameters for a specified list of neurons.

    :param matplotlib.figure.Figure fig:
        The Figure object to draw subplots on.
    :param matplotlib.gridspec.GridSpec gs:
        A GridSpec object for subplot arrangement.
    :param container.EMData emd:
        Fitted model data (posterior means).
    :param numpy.ndarray THETA:
        True parameter array of shape (T, N, N+1).
    :param list selected_neurons:
        List of neuron indices to plot.

    :returns:
        None
    """
    for idx, i in enumerate(selected_neurons):
        i=i-1
        ax = plt.subplot(gs[1 + idx // 6, idx % 6])
        for j in range(emd.N):
            ix = j + 1
            confbound = 1.96
            x_min = emd.theta_s[:, i, ix] - confbound * np.sqrt(emd.sigma_s[:, i, ix, ix])
            x_max = emd.theta_s[:, i, ix] + confbound * np.sqrt(emd.sigma_s[:, i, ix, ix])
            ax.plot(emd.theta_s[:, i, ix], alpha=0.8)
            ax.plot(THETA[:, i, ix], color='black', linestyle='--', alpha=0.8)
            ax.fill_between(range(emd.T), x_min, x_max, alpha=0.2)
        ax.set_title(r'Neuron: $\theta_{%dj,t}$' % (i + 1), fontsize=fontsize_title)
        ax.set_xlabel('Time [AU]', fontsize=fontsize_label)
        ax.set_xlim([0, emd.T])
        ax.tick_params(axis='both', labelsize=fontsize_ticklabel)

def sampling_figure(t, position, emd, THETA, T, N, gs, fig):
    """
    Plots a scatter of estimated vs. true coupling parameters at a specific time t.

    :param int t:
        Time index for which to compare parameters (0 <= t < T).
    :param int position:
        Subplot column index in the bottom row of the GridSpec.
    :param container.EMData emd:
        Fitted model data (posterior means).
    :param numpy.ndarray THETA:
        True parameter array of shape (T, N, N+1).
    :param int T:
        Number of time bins.
    :param int N:
        Number of neurons.
    :param matplotlib.gridspec.GridSpec gs:
        GridSpec object for arranging subplots.
    :param matplotlib.figure.Figure fig:
        The Figure object on which the subplot is drawn.

    :returns:
        None
    """
    theta_est = np.zeros((N, N))
    theta_true = np.zeros((N, N))
    for i in range(N):
        for j in range(1, N):
            theta_est[i, j] = emd.theta_s[t, i, j + 1]
            theta_true[i, j] = THETA[t, i, j + 1]

    ax = fig.add_subplot(gs[3, position])
    ax.set_aspect('equal', 'box')
    ax.set_title(f"$t={t}$", fontsize=fontsize_title)
    ax.tick_params(axis='both', labelsize=fontsize_ticklabel)
    if position == 0:
        ax.set_ylabel('Estimated theta', fontsize=fontsize_label)
    ax.set_xlabel('True theta', fontsize=fontsize_label)
    x = np.linspace(-2, 5)
    ax.plot(x, x, color='black')  # y = x line
    ax.scatter(theta_true, theta_est, c='blue', alpha=0.7)

def plot_results_n12(emd, THETA, spikes, N, T):
    """
    Plots the main results for the case N=12:
      - Spike rasters (Trials 1, 100, 200)
      - Time-varying coupling parameters for selected neurons
      - Scatter of estimated vs. true couplings at various time points

    :param container.EMData emd:
        Fitted model data for N=12.
    :param numpy.ndarray THETA:
        True parameter array of shape (T, N, N+1).
    :param numpy.ndarray spikes:
        A 3D binary array of shape (T, R, N).
    :param int N:
        Number of neurons (12).
    :param int T:
        Number of time bins.

    :returns:
        None
    """
    fig = plt.figure(figsize=(30, 27))
    gs = gridspec.GridSpec(4, 6)

    # A) Spike rasters
    display_neuron_image_and_spikes(spikes, gs)

    # B) Coupling parameters for selected neurons
    selected_neurons = list(range(N))
    display_selected_neuron_theta(fig, gs, emd, THETA, selected_neurons)

    # C) Scatter of estimated vs. true coupling at various times
    time_and_grid_positions = [(10, 0), (20, 1), (30, 2), (40, 3), (50, 4), (60, 5)]
    for t, n in time_and_grid_positions:
        sampling_figure(t, n, emd, THETA, T, N, gs, fig)

    # Optional labeling
    labels = ['A', 'B', 'C']
    label_positions = [
        [0.12, 0.9],   # A
        [0.12, 0.7],   # B
        [0.12, 0.27],  # C
    ]
    for label, (x, y) in zip(labels, label_positions):
        ax = fig.add_axes([x, y, .05, .05], frameon=False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.0, 0.0, label, fontsize=fontsize_ABC, fontweight='bold')

    sub_labels = [
        r'Coupling parameters, $\theta_{ij,t|T}$',
        r'Estimated vs true coupling parameters, $\theta_{ij,t}$'
    ]
    label_x_positions = [0.45, 0.45]
    label_y_positions = [0.7, 0.27]
    for x, y, label in zip(label_x_positions, label_y_positions, sub_labels):
        ax = fig.add_axes([x, y, .05, .05], frameon=False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.0, 0.0, label, fontsize=fontsize_title)

    if not os.path.exists("fig"):
        os.mkdir("fig")
    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.savefig('fig/figure12.eps', bbox_inches='tight')
    plt.savefig('fig/figure12.pdf', bbox_inches='tight')
    plt.show()

# ===================================================================
# 5) Main
# ===================================================================
def main(N):
    """
    Main function to:
      - Generate Theta
      - Generate spikes
      - Plot the results (for either N=2 or N=12)

    :param int N:
        Number of neurons. Use N=2 or N=12 to produce figures.

    :returns:
        None
    """
    np.random.seed(0)

    if N == 2:
        R = 200
        T = 400
        THETA = generate_theta(T, N)
        spikes = generate_spikes(T, R, N, THETA)
        emd = ssll_kinetic.run(spikes, max_iter=120, mstep=True)
        plot_results_n2(emd, THETA, spikes, N, T)

    elif N == 12:
        R = 200
        T = 75
        THETA = generate_theta(T, N)
        spikes = generate_spikes(T, R, N, THETA)
        emd = ssll_kinetic.run(spikes, max_iter=120, mstep=True)
        plot_results_n12(emd, THETA, spikes, N, T)
    else:
        print('No figure generation for N =', N)


if __name__ == "__main__":
    for N in [2, 12]:
        main(N)
