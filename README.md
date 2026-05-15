# Trade Execution Research

## Overview
This project implements an execution model based on the T-ACD-ARMA framework described in the reference chapter file `chap7.pdf`. The code fits an ACD(1,1) execution probability model and an ARMA(1,1)-GARCH(1,1) price density model, then computes a Table 7.3-style expected utility summary for limit order placement.

## Research source
- Purpose: provides the theoretical model and equations for the execution probability, truncated non-execution price density, and expected utility calculations.
- Key model elements from the reference:
  - ACD(1,1) with Generalized Gamma innovations for execution probability
  - ARMA-GARCH for non-execution price density
  - Utility and variance formulas for limit order execution decisions
  - Table 7.3 format for reporting results

## Dataset used
- Attached dataset folder: `Dataset/`
- Available data files:
  - `GE_10min_2006_2010_raw.csv`
  - `GE_10min_2006_2010_with_vwap.csv`
  - `GE_daily_ohlcv_vwap_2006_2010.csv`
  - `IBM_10min_2006_2010_raw.csv`
  - `IBM_10min_2006_2010_with_vwap.csv`
  - `IBM_daily_ohlcv_vwap_2006_2010.csv`
  - `MSFT_10min_2006_2010_raw.csv`
  - `MSFT_10min_2006_2010_with_vwap.csv`
  - `MSFT_daily_ohlcv_vwap_2006_2010.csv`
- Experiment datasets used in `torun/`:
  - `GE_10min_2006_2010_with_vwap.csv`
  - `IBM_10min_2006_2010_with_vwap.csv`
  - `MSFT_10min_2006_2010_with_vwap.csv`

## What the code does
- `main3.py` is the primary script for reproducing Table 7.3-style results.
- It performs the following steps:
  1. Reads a CSV dataset and prepares returns and intraday fluctuations.
  2. Fits an ACD(1,1) execution model to the fluctuation series.
  3. Fits an ARMA(1,1)-GARCH(1,1) model to price returns.
  4. Computes execution probability `P_E` and non-execution utility via a truncated price density.
  5. Prints a Table 7.3-style summary for a set of risk aversion values.

## How to run
- Default: run all files in `torun/`
  ```bash
  python main3.py
  ```
- Run a single specific CSV:
  ```bash
  python main3.py --csv torun/GE_10min_2006_2010_with_vwap.csv
  ```
- Run a specific directory:
  ```bash
  python main3.py --csv-dir torun
  ```
- Change tick size:
  ```bash
  python main3.py --csv-dir torun --tick-size 0.01
  ```

## Results on the attached datasets
### GE_10min_2006_2010_with_vwap.csv
- `P_E` = 0.79
- Utility values for risk aversion λ:
  - λ = 0.0 → `E(U)` = -0.1031, `Objective` = -0.1031
  - λ = 0.5 → `E(U)` = -0.1031, `Objective` = -0.1437
  - λ = 1.0 → `E(U)` = -0.1031, `Objective` = -0.1843
  - λ = 1.5 → `E(U)` = -0.1031, `Objective` = -0.2250
  - λ = 2.0 → `E(U)` = -0.1031, `Objective` = -0.2656

### IBM_10min_2006_2010_with_vwap.csv
- `P_E` = 1.00
- Utility values for risk aversion λ:
  - λ = 0.0 → `E(U)` = 0.01, `Objective` = 0.01
  - λ = 0.5 → `E(U)` = 0.01, `Objective` = 0.01
  - λ = 1.0 → `E(U)` = 0.01, `Objective` = 0.01
  - λ = 1.5 → `E(U)` = 0.01, `Objective` = 0.01
  - λ = 2.0 → `E(U)` = 0.01, `Objective` = 0.01

### MSFT_10min_2006_2010_with_vwap.csv
- `P_E` = 0.85
- Utility values for risk aversion λ:
  - λ = 0.0 → `E(U)` = 0.0038, `Objective` = 0.0038
  - λ = 0.5 → `E(U)` = 0.0038, `Objective` = 0.0037
  - λ = 1.0 → `E(U)` = 0.0038, `Objective` = 0.0035
  - λ = 1.5 → `E(U)` = 0.0038, `Objective` = 0.0033
  - λ = 2.0 → `E(U)` = 0.0038, `Objective` = 0.0031

## Summary
- `chap7.pdf` is the reference research source for the model and Table 7.3 methodology.
- `Dataset/` is the empirical data used to fit and validate the model.
- `main3.py` is the executable implementation that reproduces the Table 7.3-style output.
- The actual results on the `torun/` datasets are shown above.

## Learnings from running on this data
- GE showed a high execution probability (`P_E` = 0.79) but negative expected utility for the chosen limit price, indicating the model still expects a small loss under current assumptions.
- IBM produced an execution probability of `P_E` = 1.00, suggesting the limit price was effectively guaranteed to execute in the test sample and the utility estimate was positive and stable across risk aversion values.
- MSFT had a strong execution probability (`P_E` = 0.85) with slightly positive expected utility, showing that the combined ACD and ARMA-GARCH model can produce favorable limit order outcomes for some intraday datasets.
- The results highlight how different stock datasets can yield very different execution probability and utility profiles, so the model must be calibrated separately for each instrument.
- The consistent trade-off across λ shows that higher risk aversion lowers the objective value even when expected utility is unchanged, reinforcing the importance of the variance penalty in the T-ACD-ARMA framework.
