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
Inside fig1_2.py, the if __name__ == "__main__": block controls parameter loading or generation, the EM fitting process, and the plotting of Figures 1 and 2.

Figure 4
Run:
python main_kinetic/fig4.py
Similar to fig1_2.py, this script contains its own if __name__ == "__main__": block for parameter setup, data loading/generation, and final plotting of Figure 4.

Note: By default, each script generates all data and parameters from scratch, because no precomputed data is currently shared. If you would like to reproduce the results quickly, support for shared precomputed data is planned for future updates.

Overview of the Repository
This repository includes:

A state-space kinetic Ising implementation for nonstationary neuronal dynamics.

Mean-field and sampling-based methods to compute entropy flow.

An EM algorithm (or related interface) for parameter estimation from spike trains.

Utility scripts to synthesize or load spike data and model parameters.

Toggling mechanisms (USE_SAVED_...) that let you switch between loading existing data/results or generating them from scratch.

Plotting routines for comparing different methods of entropy-flow estimation.

Key Toggles (Generating vs. Loading Data)
Inside the main scripts (e.g., fig1_2.py, fig4.py), you will often see the following boolean variables. All of them are set to False by default because no precomputed data is currently shared.

USE_SAVED_THETA

False (default): Generate new parameters (THETA) and spike data based on user-defined distributions.

True: Load previously saved parameters and spike data. (Currently not supported; data sharing is under preparation.)

USE_SAVED_EMD

False (default): Run EM fitting from scratch using spike data.

True: Load a previously saved EM fitting object (emd). (Currently not supported.)

USE_SAVED_SAMPLING

False (default): Generate new spike data and compute sampling-based entropy flow.

True: Load previously computed sampling-based entropy-flow results. (Currently not supported.)

Since all three toggles are False by default, the script will perform full data generation, EM fitting, and entropy computation. This can require up to 90 minutes on standard hardware using the default settings.

Parameter Setup
Within each main script (e.g., fig4.py), you can modify simulation parameters such as:

python

N = 80          # Number of neurons
T = 150         # Number of time steps
R = 500         # Number of repeated experimental trials 
R_sampling = 500  # Number of trials for sampling-based entropy flow

coupling_mu = 5 / N
coupling_sigma = 30
coupling_alpha = 0.1 * N

field_mu = -3
field_sigma = 50
field_alpha = 1
Reducing N, T, or R will shorten the computation time.
Increasing them will significantly increase runtime.

コピーする
