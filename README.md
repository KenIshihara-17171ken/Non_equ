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

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15220108.svg)](https://doi.org/10.5281/zenodo.15220108)

**How to Download and Use the Data:**

1. **Download the Dataset:**
   - Click the DOI badge above or visit [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15220108.svg)](https://doi.org/10.5281/zenodo.15220108)to navigate to the Zenodo record.
   - On the Zenodo page, download the `saved_data.zip` file (approximately 5.0 GB).

2. **Extract the Dataset:**
   - Once downloaded, extract the `saved_data.zip` file.
   - **Important:** Ensure that the extracted folder is placed under the `main_kinetic` directory of this repository, such that the directory structure becomes:
     ```
     main_kinetic/
       └── saved_data/
            ├── data_emd/
            ├── data_sampling_entropy_flow/
            └── data_theta_spike/
     ```

3. **Run the Code:**
   - After placing the extracted `saved_data` folder in the correct location, you can run the provided Python scripts (e.g.,`python main_kinetic/fig4.py`) as described in the documentation.

Following these steps will allow you to access the complete dataset hosted on Zenodo and successfully execute the analysis code in this repository.


Run:
```bash
python main_kinetic/fig4.py
```
Similar to fig1_2.py, this script contains its own if __name__ == "__main__": block for parameter setup, data loading/generation, and final plotting of Figure 4.

### Overview of the Repository
This repository includes:

- A state-space kinetic Ising implementation for nonstationary neuronal dynamics.

- Mean-field and sampling-based methods to compute entropy flow.

- An EM algorithm (or related interface) for parameter estimation from spike trains.

- Utility scripts to synthesize or load spike data and model parameters.

- Toggling mechanisms (USE_SAVED_...) that let you switch between loading existing data/results or generating them from scratch.

- Plotting routines for comparing different methods of entropy-flow estimation.
