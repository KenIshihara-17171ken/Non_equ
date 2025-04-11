"""
This code implements the state-space kinetic Ising model described in:
Ken Ishihara, Hideaki Shimazaki. *State-space kinetic Ising model reveals task-dependent entropy flow in sparsely active nonequilibrium neuronal dynamics*. (2025) arXiv:2502.15440

The implementation extends existing libraries available at:
- https://github.com/christiando/ssll_lib.git
- https://github.com/shimazaki/dynamic_corr

This implementation also incorporates and adapts mean-field approximation techniques based on:
- https://github.com/MiguelAguilera/kinetic-Plefka-expansions.git

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

################################################################
# Sigmoid and entropy-related utilities
################################################################
def sigmoid(a):
    """
    Standard sigmoid function: sigmoid(a) = 1 / (1 + exp(-a)).

    Parameters
    ----------
    a : float or np.ndarray
        Input scalar or array.

    Returns
    -------
    float or np.ndarray
        Sigmoid-transformed values (element-wise if `a` is an array).
    """
    return 1 / (1 + np.exp(-a))


def chi(a):
    """
    Computes chi(a) = -[ p*log(p) + (1 - p)*log(1 - p) ],
    where p = 1 / (1 + exp(a)).

    Parameters
    ----------
    a : float or np.ndarray
        Input scalar or array.

    Returns
    -------
    float or np.ndarray
        The value of chi(a) (element-wise if `a` is an array).
    """
    oe = 1 / (1 + np.exp(a))
    return -(oe * np.log(oe) + (1 - oe) * np.log(1 - oe))

################################################################
# 1D and 2D Gaussian integration
################################################################
def integrate_1DGaussian(f, args=(), Nint=100):
    """
    Performs a 1D Gaussian numerical integration over the interval [-4, 4].
    Uses a uniform grid of Nint points from -4 to +4 (scaled from [-1,1]*4).

    Parameters
    ----------
    f : callable
        A function of the form f(x, *args) to be integrated.
    args : tuple, optional
        Additional arguments to pass to f.
    Nint : int, optional
        Number of integration points (default 100).

    Returns
    -------
    float
        The approximate integral of f(x) over x in [-4, 4].
    """
    x = np.linspace(-1, 1, Nint) * 4
    dx = x[1] - x[0]
    return np.sum(f(x, *args)) * dx


def integrate_2DGaussian(f, args=(), Nx=20, Ny=20):
    """
    Performs a 2D discrete approximation to a Gaussian integral
    over the range [-4,4] x [-4,4], using Nx x Ny grid points.

    Parameters
    ----------
    f : callable
        A function of the form f(px, nx, *args) to be integrated in 2D.
    args : tuple, optional
        Additional arguments to pass to f.
    Nx : int, optional
        Number of grid divisions along the x-direction (default 20).
    Ny : int, optional
        Number of grid divisions along the y-direction (default 20).

    Returns
    -------
    float
        The approximate 2D integral of f over the domain.
    """
    p = np.linspace(-1, 1, Nx) * 4
    n = np.linspace(-1, 1, Ny) * 4
    P, N = np.meshgrid(p, n)
    val = f(P, N, *args)
    return np.sum(val) * (p[1] - p[0]) * (n[1] - n[0])

################################################################
# Integrand functions used in forward/reverse entropy calculations
################################################################
def dT_s(x, g, D):
    """
    Integrand for forward-entropy calculations (chi(...) in a 1D Gaussian integral).

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the integrand.
    g : float
        Local field (e.g., H[i] + sum_j(J[i,j]*m_prev[j])) for a neuron.
    D : float
        Variance term in the mean-field approximation.

    Returns
    -------
    np.ndarray
        The integrand values at each point x.
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * chi(g + x * np.sqrt(D))


def dT1(x, g, D):
    """
    Integrand for updating spike probability (mean spikes) with a sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the integrand.
    g : float
        Local field.
    D : float
        Variance term in the mean-field approximation.

    Returns
    -------
    np.ndarray
        The integrand values, using sigmoid(g + x * sqrt(D)).
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * sigmoid(g + x * np.sqrt(D))


def dT_sr_0(x, g, D):
    """
    Reverse-entropy integrand for the case of spike=0.

    Effectively:
      - log(1 + exp(g + x*sqrt(D))) * Gaussian_weight
    """
    A = 0.0
    B = -np.log(1 + np.exp(g + x * np.sqrt(D)))
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * (A + B)


def dT_sr_1(x, g, D):
    """
    Reverse-entropy integrand for the case of spike=1.
    """
    A = (g + x * np.sqrt(D))
    B = -np.log(1 + np.exp(g + x * np.sqrt(D)))
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * (A + B)


def dT_sr_h(x, g, D):
    """
    Integrand for partial calculations (e.g., update_S_t / update_S_re_t).
    Often used for the 'h' term in a log-likelihood approach.
    """
    A = (g + x * np.sqrt(D))
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * A


def dT_sr_psi(x, g, D):
    """
    Integrand for partial calculations involving log(1 + exp(...)).
    Used in update_S_t / update_S_re_t.
    """
    A = (g + x * np.sqrt(D))
    B = np.log(1 + np.exp(A))
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * B


def dT1_1(x, g, D):
    """
    Integrand used in update_D_P_t1_o1 for terms of the form (1 - sigmoid(...)^2).

    Parameters
    ----------
    x : np.ndarray
        Integration points.
    g : float
        Local field.
    D : float
        Variance term.

    Returns
    -------
    np.ndarray
        The integrand values for the expression (1 - sigmoid(... )^2).
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) \
           * (1 - sigmoid(g + x * np.sqrt(D))**2)


def dT2_rot(p, n, gx, gy, Dx, Dy, rho):
    """
    Example integrand used for 2D Gaussian integration with correlation (rho).
    Includes specialized handling for p=None or n=None cases.
    """
    if n is None:
        # One-dimensional approximation using p
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * p**2) \
               * chi(gx + p * np.sqrt(1 + rho) * np.sqrt(Dx / 2)) \
               * chi(gy + p * np.sqrt(1 + rho) * np.sqrt(Dy / 2))
    elif p is None:
        # One-dimensional approximation using n
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * p**2) \
               * chi(gx + n * np.sqrt(1 - rho) * np.sqrt(Dx / 2)) \
               * chi(gy - n * np.sqrt(1 - rho) * np.sqrt(Dy / 2))
    else:
        # Full 2D integration
        return (1 / (2 * np.pi)) * np.exp(-0.5 * (p**2 + n**2)) \
               * chi(gx + (p * np.sqrt(1 + rho) + n * np.sqrt(1 - rho)) * np.sqrt(Dx / 2)) \
               * chi(gy + (p * np.sqrt(1 + rho) - n * np.sqrt(1 - rho)) * np.sqrt(Dy / 2))

################################################################
# Major functions for forward/reverse entropy and mean spikes updates
################################################################

def update_S(H, J, m_p):
    """
    Computes the forward entropy:
      For each i:  entropy_flow_forward_i[i] = ∫ dT_s(...) over x, 
      then sum over i -> entropy_flow_forward.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
        External fields for each neuron.
    J : np.ndarray, shape (N, N)
        Coupling matrix.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes/spike probabilities.

    Returns
    -------
    entropy_flow_forward : float
        The total forward entropy (sum over neurons).
    entropy_flow_forward_i : np.ndarray, shape (N,)
        Forward entropy contribution per neuron.
    """
    size = len(H)
    entropy_flow_forward_i = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        entropy_flow_forward_i[i] = integrate_1DGaussian(dT_s, (g[i], D[i]))
    entropy_flow_forward = np.sum(entropy_flow_forward_i)
    return entropy_flow_forward, entropy_flow_forward_i


def update_S_re(H, J, m, m_p):
    """
    Computes the reverse entropy ("re" version):
      For each i:
        phi_0[i] = ∫ dT_sr_0(...),
        phi_1[i] = ∫ dT_sr_1(...),
        entropy_flow_reverse_i[i] = -( m_p[i]*phi_1[i] + (1 - m_p[i])*phi_0[i] ).
      entropy_flow_reverse = sum over i.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
        External fields.
    J : np.ndarray, shape (N, N)
        Coupling matrix.
    m : np.ndarray, shape (N,)
        Current mean spikes/spike probabilities.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes/spike probabilities.

    Returns
    -------
    entropy_flow_reverse : float
        The total reverse entropy.
    entropy_flow_reverse_i : np.ndarray, shape (N,)
        Reverse entropy contribution per neuron.
    """
    size = len(H)
    phi_0 = np.zeros(size)
    phi_1 = np.zeros(size)
    entropy_flow_reverse_i = np.zeros(size)
    g = H + np.dot(J, m)
    D = np.dot(J**2, m * (1 - m))
    for i in range(size):
        phi_0[i] = integrate_1DGaussian(dT_sr_0, (g[i], D[i]))
        phi_1[i] = integrate_1DGaussian(dT_sr_1, (g[i], D[i]))
        entropy_flow_reverse_i[i] = -(m_p[i] * phi_1[i] + (1 - m_p[i]) * phi_0[i])
    entropy_flow_reverse = np.sum(entropy_flow_reverse_i)
    return entropy_flow_reverse, entropy_flow_reverse_i


def update_S_t(H, J, m, m_p):
    """
    Alternative forward-entropy calculation using h and psi terms:
      For each i, 
        h[i]   = ∫ dT_sr_h(...),
        psi[i] = ∫ dT_sr_psi(...),
        S[i]   = -( m[i]*h[i] - psi[i] ).
      Summation returned as total.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
    J : np.ndarray, shape (N, N)
    m : np.ndarray, shape (N,)
        Current mean spikes.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    float
        The total forward entropy computed via h and psi integrals.
    """
    size = len(H)
    S = np.zeros(size)
    h_vals = np.zeros(size)
    psi_vals = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        h_vals[i] = integrate_1DGaussian(dT_sr_h, (g[i], D[i]))
        psi_vals[i] = integrate_1DGaussian(dT_sr_psi, (g[i], D[i]))
        S[i] = -(m[i] * h_vals[i] - psi_vals[i])
    return np.sum(S)


def update_S_re_t(H, J, m, m_p):
    """
    Alternative reverse-entropy calculation, similar to update_S_t but reversed roles.

    Parameters
    ----------
    H, J, m, m_p : same shapes as above

    Returns
    -------
    float
        The total reverse entropy using h and psi integrals in a reversed manner.
    """
    size = len(H)
    h_vals = np.zeros(size)
    psi_vals = np.zeros(size)
    S = np.zeros(size)
    g = H + np.dot(J, m)
    D = np.dot(J**2, m * (1 - m))
    for i in range(size):
        h_vals[i] = integrate_1DGaussian(dT_sr_h, (g[i], D[i]))
        psi_vals[i] = integrate_1DGaussian(dT_sr_psi, (g[i], D[i]))
        S[i] = -(m_p[i] * h_vals[i] - psi_vals[i])
    return np.sum(S)


def update_S_ind(H, m, m_p):
    """
    Computes an "indirect" term: sum over i of [ H[i]*(m[i] - m_p[i]) ].

    Parameters
    ----------
    H : np.ndarray, shape (N,)
    m : np.ndarray, shape (N,)
        Current mean spikes.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    float
        The sum of H[i]*(m[i] - m_p[i]) over i.
    """
    size = len(H)
    S_ind = np.zeros(size)
    for i in range(size):
        S_ind[i] = H[i] * (m[i] - m_p[i])
    return np.sum(S_ind)


def update_m_P_t1_o1(H, J, m_p):
    """
    Mean-field mean spikes update: 
      m[i] = ∫ (1/sqrt(2π)) exp(-x^2/2)*sigmoid(g + x*sqrt(D)) dx,
    where g = H + Σ_j J*m_p, and D = Σ_j (J^2)*m_p*(1-m_p).

    Parameters
    ----------
    H : np.ndarray, shape (N,)
    J : np.ndarray, shape (N, N)
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    np.ndarray, shape (N,)
        Updated mean spikes m.
    """
    size = len(H)
    m = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        m[i] = integrate_1DGaussian(dT1, (g[i], D[i]))
    return m


############################################################

def computation_m(a, m_p):
    """
    Calculates mean-field mean spikes m from parameter array `a` (shape (N, N+1)) 
    and previous mean spikes m_p.

    a[:,0] = external field H,  a[:,1:] = coupling J.

    Parameters
    ----------
    a : np.ndarray, shape (N, N+1)
        Parameter array where a[i,0] = H[i], a[i,1:] = J[i,:].
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    m : np.ndarray, shape (N,)
        Updated mean spikes (or spike probability).
    """
    h = a[:, 0]
    j = np.delete(a, 0, 1)
    H = h
    J = j
    m = update_m_P_t1_o1(H, J, m_p)
    # Alternatively: m = sigmoid(H + J.dot(m_p))
    return m


def Dissipation_en(a, m, m_p):
    """
    Computes the forward and reverse entropy flows, as well as their difference,
    returning both total and per-neuron values.

    Specifically:
      - Forward: (entropy_flow_forward, entropy_flow_forward_i)
      - Reverse: (entropy_flow_reverse, entropy_flow_reverse_i)
      - Net flow = ( -entropy_flow_forward + entropy_flow_reverse ) at total and nodewise levels.

    Parameters
    ----------
    a : np.ndarray, shape (N, N+1)
        Parameter array, where a[i,0] = H[i], a[i,1:] = J[i,:].
    m : np.ndarray, shape (N,)
        Current mean spikes (or spike probability).
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    entropy_flow_forward : float
        Total forward entropy (sum over all neurons).
    entropy_flow_reverse : float
        Total reverse entropy (sum over all neurons).
    net_flow : float
        Overall net flow = ( -entropy_flow_forward + entropy_flow_reverse ).
    entropy_flow_forward_i : np.ndarray, shape (N,)
        Forward entropy contribution for each neuron.
    entropy_flow_reverse_i : np.ndarray, shape (N,)
        Reverse entropy contribution for each neuron.
    net_node : np.ndarray, shape (N,)
        Per-neuron net flow = ( -entropy_flow_forward_i + entropy_flow_reverse_i ).
    """
    # Forward
    h = a[:, 0]
    j = np.delete(a, 0, 1)
    H = h
    J = j
    entropy_flow_forward_t, entropy_flow_forward_t_i = update_S(H, J, m_p)

    # Recompute m for reverse step
    m_new = update_m_P_t1_o1(H, J, m_p)

    # Reverse
    entropy_flow_reverse_t, entropy_flow_reverse_t_i = update_S_re(H, J, m_new, m_p)

    entropy_flow_t = -entropy_flow_forward_t + entropy_flow_reverse_t
    entropy_flow_t_i = -entropy_flow_forward_t_i + entropy_flow_reverse_t_i

    return (
        entropy_flow_forward_t,
        entropy_flow_reverse_t,
        entropy_flow_t,
        entropy_flow_forward_t_i,
        entropy_flow_reverse_t_i,
        entropy_flow_t_i
    )


################################################################
# Additional functions (D_function, C_function)
################################################################

def D_function(a, m_p, C):
    """
    Example function to update a 'D(t)' using the parameter array a, previous
    mean spikes m_p, and some correlation matrix C.

    Parameters
    ----------
    a : np.ndarray, shape (N, N+1)
        Parameter array for the current time step.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.
    C : np.ndarray
        Some correlation matrix or related structure.

    Returns
    -------
    d : np.ndarray
        Updated D matrix after performing the relevant integrals and transformations.
    """
    h = a[:, 0]
    j = np.delete(a, 0, 1)
    H = h
    J = j
    c_1 = C
    d = update_D_P_t1_o1(H, J, m_p, c_1)
    return d


def update_D_P_t1_o1(H, J, m_p, C_p):
    """
    Sub-function to compute and update a 'D' matrix or similar quantity.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
    J : np.ndarray, shape (N, N)
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.
    C_p : np.ndarray
        Prior correlation structure.

    Returns
    -------
    d : np.ndarray
        The updated result based on integrating dT1_1(...) and applying J, C_p.
    """
    size = len(H)
    a_vals = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    for i in range(size):
        a_vals[i] = integrate_1DGaussian(dT1_1, (g[i], D[i]))
    return np.einsum('i,ij,jl->il', a_vals, J, C_p)


def C_function(a, m_p):
    """
    Example function to update a correlation matrix C(t).

    Parameters
    ----------
    a : np.ndarray, shape (N, N+1)
        Parameter array for the current time step.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    c : np.ndarray, shape (N, N)
        Updated correlation matrix.
    """
    h = a[:, 0]
    j = np.delete(a, 0, 1)
    H = h
    J = j
    m = update_m_P_t1_o1(H, J, m_p)
    c = update_C_P_t1_o1(H, J, m, m_p)
    print(len(c))
    return c


def update_C_P_t1_o1(H, J, m, m_p):
    """
    Computes or updates a correlation matrix C given the current parameters
    and mean spikess.

    Parameters
    ----------
    H : np.ndarray, shape (N,)
    J : np.ndarray, shape (N, N)
    m : np.ndarray, shape (N,)
        Current mean spikes.
    m_p : np.ndarray, shape (N,)
        Previous mean spikes.

    Returns
    -------
    C : np.ndarray, shape (N, N)
        The updated correlation matrix.
    """
    size = len(H)
    C = np.zeros((size, size))
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, m_p * (1 - m_p))
    inv_D = np.zeros(size)
    inv_D[D > 0] = 1 / D[D > 0]
    rho = np.einsum('i,k,ij,kj,j->ik',
                    np.sqrt(inv_D),
                    np.sqrt(inv_D),
                    J,
                    J,
                    m_p * (1 - m_p),
                    optimize=True)
    for i in range(size):
        C[i, i] = m_p[i] * (1 - m_p[i])
        for j2 in range(i + 1, size):
            if rho[i, j2] > (1 - 1E5):
                # Use 1D integral approximation
                C[i, j2] = integrate_1DGaussian(dT2_rot, (None, g[i], g[j2], D[i], D[j2], rho[i, j2])) \
                           - m[i] * m[j2]
            else:
                # Use full 2D integral
                C[i, j2] = integrate_2DGaussian(dT2_rot, (g[i], g[j2], D[i], D[j2], rho[i, j2])) \
                           - m[i] * m[j2]
            C[j2, i] = C[i, j2]
    return C
