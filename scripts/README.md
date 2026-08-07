# Scripts Directory

This directory contains utility scripts that execute supplemental tasks outside of the core data processing and modeling pipelines.

## Robustness Checks

- **`run_rq_todos.py`**: A standalone script used to run alternative econometric re-estimates (robustness checks) for the main research questions. It reads the pre-assembled panel datasets from the `results/` directory and performs the following operations:
  - **RQ1 Re-estimate (`log_volume`)**: Re-runs the RQ1 (stance shift) panel regression using log-normalized article volume (`log(1 + volume)`) instead of raw article volume to ensure the volume control is robust to outliers.
  - **RQ2 Re-estimate (`two_way_fe`)**: Re-runs the RQ2 (returns shift) panel regression employing two-way fixed effects (firm and time/date) to more rigorously absorb common unobserved shocks.
  
The outputs from these alternative models are saved back into the `results/rq1/log_volume/` and `results/rq2/two_way_fe/` directories, respectively.
