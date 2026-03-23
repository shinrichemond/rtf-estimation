## rtf-estimation/simulation Overview

This directory contains the implementation and data generation pipeline for simulating the p-THYROSIM model and constructing datasets used for RTF estimation.

### Core Files
- **Clean_Code_LT43DosingApp.ipynb**  
  Original reference implementation of the p-THYROSIM model. Serves as the baseline formulation and validation source for subsequent code.

- **pthyrosim_model.py**  
  Python port of the p-THYROSIM model. Implements the full simulation pipeline, including hormone dynamics, dosing inputs, and time-series generation. This file is the core engine used by all downstream scripts.

- **sweep.py**  
  Performs parameter sweeps over patient characteristics (height, weight, sex), dosing regimens (LT4, LT3), and RTF values. Generates a structured simulation grid used for estimator development.

- **createData.py**  
  Constructs synthetic patient datasets by sampling single timepoint observations from simulated hormone trajectories. Introduces realistic temporal variability to mimic clinical measurements.

- **counterexample.ipynb**  
  Demonstrates failure modes of naive RTF estimation under exogenous dosing. Provides exploratory analysis and supporting visualizations.

### Datasets
- **thyrosim_cut_dataset_v2.csv**  
  Precomputed simulation grid containing steady-state hormone summaries across parameter combinations. Used for grid-based and optimization-based estimators.

- **thyrosim_sample_data.csv**  
  Synthetic patient dataset containing randomly sampled timepoint measurements and corresponding ground-truth parameters.

### Suggested Workflow

1. Use `pthyrosim_model.py` to simulate hormone dynamics.  
2. Run `sweep.py` to generate the full parameter grid dataset.  
3. Use `createData.py` to construct sampled clinical observations.  
4. Apply estimation methods using the generated datasets.  
5. Refer to `counterexample.ipynb` for known failure cases.