# State-Space Kinetic Ising Model

This repository provides Python code for analyzing **non-stationary and nonequilibrium** neuronal spiking activity using a state-space kinetic Ising model. The model captures time-varying firing rates and causal couplings, enabling the estimation of **time-asymmetric** (nonequilibrium) dynamics such as entropy flow.

## Requirements

- **Python version**: Tested on **Python 3.8.10** (should work on 3.8+)
- Recommended libraries:
  - **numpy**
  - **matplotlib**
  - **scipy**
  - **joblib** or **numba** (optional; can speed up certain computations)

For further details, please refer to the preprint:  
[https://arxiv.org/abs/2502.15440](https://arxiv.org/abs/2502.15440)

## How to Reproduce the Figures

1. **Figure 1 and 2**  
   - Run `python main_kinetic/fig1_2.py`.  
   - Inside `fig1_2.py`, the `if __name__ == "__main__":` block controls parameter loading or generation, the EM fitting process, and the plotting of Figures 1 and 2.

2. **Figure 4**  
   - Run `python main_kinetic/fig4.py`.  
   - Similar to `fig1_2.py`, the script contains its own `if __name__ == "__main__":` block for parameter setup, data loading/generation, and final plotting of Figure 4.

By default, each script attempts to load previously saved data/results to **quickly** reproduce the figures. If you prefer to generate new data or re-run the EM fitting, set the relevant toggles (e.g., `USE_SAVED_THETA`, `USE_SAVED_EMD`, `USE_SAVED_SAMPLING`) to **False** in the script.

## Overview of the Repository

This repository includes:

- A **state-space kinetic Ising** implementation for nonstationary neuronal dynamics.
- **Mean-field** and **sampling-based** methods to compute entropy flow.
- An **EM algorithm** (or related interface) for parameter estimation from spike trains.
- Utility scripts to synthesize or load **spike data** and **model parameters**.
- Toggling mechanisms (`USE_SAVED_...`) that let you switch between loading existing data/results or generating them from scratch.
- Plotting routines for comparing different methods of entropy-flow estimation.

## Key Toggles (Generating vs. Loading Data)

Inside the main scripts (e.g.,`fig4.py`), you will often see three boolean variables:

1. **`USE_SAVED_THETA`**  
   - `True`: Load previously saved `THETA` (and spike data).  
   - `False`: Generate new parameters and spikes based on user-defined distributions.

2. **`USE_SAVED_EMD`**  
   - `True`: Load a previously saved EM fitting object (`emd`).  
   - `False`: Perform an EM fitting from scratch on the spike data.

3. **`USE_SAVED_SAMPLING`**  
   - `True`: Load previously computed sampling-based entropy-flow results.  
   - `False`: Generate new spike data and compute sampling-based entropy flow.

Set all three to **True** for the fastest reproduction of the paper's figures (assuming the saved files exist). If you wish to experiment with different numbers of neurons (**N**), time steps (**T**), or coupling/field parameters, set one or more toggles to **False** so that the script will regenerate the data and run the computations.

## Parameter Setup

Within each main script (e.g., `fig4.py`), you can modify parameters such as:

```python
N = 80          # Number of neurons
T = 150         # Number of time steps
R = 500         # Trials for fitting
R_sampling = 500

coupling_mu = 5 / N
coupling_sigma = 30
coupling_alpha = 0.1 * N

field_mu = -3
field_sigma = 50
field_alpha = 1
