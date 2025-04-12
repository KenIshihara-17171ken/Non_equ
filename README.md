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
bash
python main_kinetic/fig1_2.py

Inside fig1_2.py, the if __name__ == "__main__": block controls parameter loading or generation, the EM fitting process, and the plotting of Figures 1 and 2.
