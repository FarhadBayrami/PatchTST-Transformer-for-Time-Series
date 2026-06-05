# Sorbolo PatchTST — River Discharge Forecasting

PatchTST deep learning model for one-step-ahead river discharge forecasting.
Trained on the Sorbolo basin (Po River, Italy) using hourly hydro-meteorological data.

## Structure

\\\
sorbolo_patchtst/
+-- 1__prapare_data.py     # data loading, scaling, sequence creation
+-- 2__train.py            # PatchTST model definition and training loop
+-- 3__evaluate.py         # inference, metrics, plots
+-- 4_visualize_tensors.py
+-- LSTM.ini               # all configuration (paths, hyperparameters)
+-- data/
¦   +-- raw/               # input CSV (not tracked)
¦   +-- processed/         # .npy arrays (not tracked)
+-- models/                # model weights (not tracked)
+-- outputs/
    +-- plots/             # all output plots (tracked)
\\\

## Model

- Architecture: PatchTST with dual output heads (log-space + magnitude)
- Input: 60-step sliding window of hydro-meteorological features
- Output: one-step-ahead QM (measured discharge) in m3/s
- Metrics: NSE, RMSE, MAE, PBIAS

## Usage

1. Place raw data CSV in data/raw/
2. Edit paths in LSTM.ini
3. Run scripts in order:

   python 1__prapare_data.py
   python 2__train.py
   python 3__evaluate.py
