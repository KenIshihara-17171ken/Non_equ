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

### Figure 1 and 2

Run:
```bash
python main_kinetic/fig1_2.py
```
Inside fig1_2.py, the if __name__ == "__main__": block controls parameter loading or generation, the EM fitting process, and the plotting of Figures 1 and 2.


### Figure 4
  ## Data Availability

Due to the large size of the dataset, the complete saved data is not hosted on GitHub. Instead, it is archived on Zenodo for long-term preservation and easy citation. You can download the data using the following link:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15205014.svg)](https://doi.org/10.5281/zenodo.15205014)

**Citation:**

ISHIHARA, K. (2025). State-Space Kinetic Ising Model Code and Data for Neuronal Spiking Analysis. Zenodo. https://doi.org/10.5281/zenodo.15205014


Run:
```bash
python main_kinetic/fig4.py
```
Similar to fig1_2.py, this script contains its own if __name__ == "__main__": block for parameter setup, data loading/generation, and final plotting of Figure 4.

Note: By default, each script generates all data and parameters from scratch, because no precomputed data is currently shared. If you would like to reproduce the results quickly, support for shared precomputed data is planned for future updates.

### Overview of the Repository
This repository includes:

- A state-space kinetic Ising implementation for nonstationary neuronal dynamics.

- Mean-field and sampling-based methods to compute entropy flow.

- An EM algorithm (or related interface) for parameter estimation from spike trains.

- Utility scripts to synthesize or load spike data and model parameters.

- Toggling mechanisms (USE_SAVED_...) that let you switch between loading existing data/results or generating them from scratch.

- Plotting routines for comparing different methods of entropy-flow estimation.
