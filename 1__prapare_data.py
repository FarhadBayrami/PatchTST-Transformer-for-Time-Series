"""
Snippet 1 — Data Preparation  (v5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Changes vs v4:
  • Q (TOPKAPI) is now saved alongside QM for the test set AND the
    full post-2014 dataset so the evaluation script can plot all
    three series (QM observed, QM predicted, Q TOPKAPI) for comparison.
  • Whole-dataset arrays (X_all, Y_all) are now also saved so the
    model can predict across the entire 2014-2024 period.
  • Everything else (split, features, scalers, lags) is identical to v4.
"""

import os, pickle, configparser
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── config ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_cfg  = configparser.ConfigParser()
_cfg.read(os.path.join(_HERE, "LSTM.ini"))
_pt   = _cfg["patchtst"]

RAW_CSV = _pt["raw_csv"]
OUT_DIR = _pt["data_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

# ── window config (keep in sync with 2_train.py) ──────────────────────────────
SEQ_LEN   = _pt.getint("seq_len")
PATCH_LEN = _pt.getint("patch_len")
STRIDE    = _pt.getint("stride")

# ── 1. load & parse ───────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV, sep=";")
df["Time"] = pd.to_datetime(df["Time"], format="%d/%m/%Y %H:%M")
df = df.sort_values("Time").reset_index(drop=True)

# ── 2. filter date ────────────────────────────────────────────────────────────
df = df[df["Time"] >= _pt["start_date"]].reset_index(drop=True)
print(f"After date filter    : {len(df)} rows")

# ── 3. drop missing QM rows ───────────────────────────────────────────────────
df = df[df["QM"] != -9999.0].reset_index(drop=True)
print(f"After QM clean       : {len(df)} rows")

# ── 4. keep Q (TOPKAPI) aside BEFORE dropping it from features ───────────────
#   Q is forbidden as a model INPUT but we DO want it for comparison plots.
#   We carry it as a separate column through the whole pipeline and save it.
df_Q = df[["Time", "Q"]].copy()   # full Q series (2014-2024), aligned with df

# ── 5. drop unwanted feature columns ─────────────────────────────────────────
df = df.drop(columns=["Q", "Deep", "DeepSat", "Inf2Surf", "EnSnow", "Surf"])

# ── 6. add lagged QM features ─────────────────────────────────────────────────
for lag in [1, 2, 3, 7, 14, 21]:
    df[f"QM_lag{lag}"] = df["QM"].shift(lag)
df = df.dropna().reset_index(drop=True)
# realign df_Q with df after dropna (first 21 rows dropped)
df_Q = df_Q[df_Q["Time"].isin(df["Time"])].reset_index(drop=True)
print(f"After lag creation   : {len(df)} rows  (df_Q rows: {len(df_Q)})")

# ── 7. define features and targets ───────────────────────────────────────────
EXCLUDE      = {"Time", "QM"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 8. YEAR-BASED SPLIT ───────────────────────────────────────────────────────
# v6 change: 2023 moved from val into training.
# Reason: val (2022-2023) had QM_max=134.9 but test (2024) hit 237.7 m3/s.
# The model never learned from events close in distribution to 2024.
# 2024 is split: Jan-Jun = validation (early stopping), Jul-Dec = test.
# Temporal order is strictly preserved — no future leakage.
train_mask = df["Time"].dt.year <= 2023
val_mask   = (df["Time"].dt.year == 2024) & (df["Time"].dt.month <= 6)
test_mask  = (df["Time"].dt.year == 2024) & (df["Time"].dt.month >= 7)

df_train = df[train_mask].copy().reset_index(drop=True)
df_val   = df[val_mask].copy().reset_index(drop=True)
df_test  = df[test_mask].copy().reset_index(drop=True)

# Q aligned to each split
Q_train = df_Q[train_mask.values].reset_index(drop=True)
Q_val   = df_Q[val_mask.values].reset_index(drop=True)
Q_test  = df_Q[test_mask.values].reset_index(drop=True)

for name, d in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
    print(f"  {name}: {len(d):4d} rows  "
          f"{d['Time'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{d['Time'].iloc[-1].strftime('%Y-%m-%d')}  "
          f"QM mean={d['QM'].mean():.1f}  max={d['QM'].max():.1f}  "
          f"n>50={(d['QM']>50).sum()}")

# ── 9. fit scalers on TRAIN only ─────────────────────────────────────────────
QM_TRAIN_MAX = df_train["QM"].max()
print(f"\nQM_TRAIN_MAX : {QM_TRAIN_MAX:.2f}")

scaler_X  = StandardScaler()
scaler_y1 = StandardScaler()

for d in [df_train, df_val, df_test, df]:
    d["QM_log"]  = np.log1p(d["QM"])
    d["QM_norm"] = d["QM"] / QM_TRAIN_MAX

X_tr_sc  = scaler_X.fit_transform(df_train[FEATURE_COLS])
X_va_sc  = scaler_X.transform(df_val[FEATURE_COLS])
X_te_sc  = scaler_X.transform(df_test[FEATURE_COLS])
X_all_sc = scaler_X.transform(df[FEATURE_COLS])       # whole dataset

y1_tr  = scaler_y1.fit_transform(df_train[["QM_log"]])
y1_va  = scaler_y1.transform(df_val[["QM_log"]])
y1_te  = scaler_y1.transform(df_test[["QM_log"]])
y1_all = scaler_y1.transform(df[["QM_log"]])

y2_tr  = df_train[["QM_norm"]].values.astype(np.float32)
y2_va  = df_val[["QM_norm"]].values.astype(np.float32)
y2_te  = df_test[["QM_norm"]].values.astype(np.float32)
y2_all = df[["QM_norm"]].values.astype(np.float32)

Y_tr  = np.hstack([y1_tr,  y2_tr ]).astype(np.float32)
Y_va  = np.hstack([y1_va,  y2_va ]).astype(np.float32)
Y_te  = np.hstack([y1_te,  y2_te ]).astype(np.float32)
Y_all = np.hstack([y1_all, y2_all]).astype(np.float32)

# ── 10. sliding-window sequences ─────────────────────────────────────────────
def make_sequences(X, Y):
    Xs, Ys = [], []
    for i in range(len(X) - SEQ_LEN):
        Xs.append(X[i : i + SEQ_LEN])
        Ys.append(Y[i + SEQ_LEN])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

X_tr,  Y_tr_seq  = make_sequences(X_tr_sc,  Y_tr)

# ── 10b. flood sequence augmentation ────────────────────────────────────────
# Physically repeat sliding-window sequences where QM_norm > threshold so the
# model sees peak events proportionally more during training.  This is more
# aggressive than the WeightedRandomSampler in 2_train.py (which only changes
# sampling probability); this actually multiplies the data.
# threshold=0.155 ~ QM > 50 m3/s  (50 / QM_TRAIN_MAX=323.6)
# repeat=8 means flood windows appear 8x more → floods go from ~3% to ~20%
_flood_thresh = 50.0 / QM_TRAIN_MAX
_flood_mask   = Y_tr_seq[:, 1] > _flood_thresh
_n_floods     = _flood_mask.sum()
X_tr_aug = np.concatenate([X_tr] + [X_tr[_flood_mask]] * 8, axis=0)
Y_tr_aug = np.concatenate([Y_tr_seq] + [Y_tr_seq[_flood_mask]] * 8, axis=0)
_idx     = np.random.default_rng(42).permutation(len(X_tr_aug))
X_tr     = X_tr_aug[_idx]
Y_tr_seq = Y_tr_aug[_idx]
print(f"Flood augmentation   : {_n_floods} flood windows x8 → ",
      f"train size {len(X_tr)} (was {len(X_tr_aug) - _n_floods*8})")
X_va,  Y_va_seq  = make_sequences(X_va_sc,  Y_va)
X_te,  Y_te_seq  = make_sequences(X_te_sc,  Y_te)
X_all, Y_all_seq = make_sequences(X_all_sc, Y_all)

n_patches = (SEQ_LEN - PATCH_LEN) // STRIDE + 1
print(f"\nX_train : {X_tr.shape}   Y_train : {Y_tr_seq.shape}")
print(f"X_all   : {X_all.shape}  Y_all   : {Y_all_seq.shape}")
print(f"Patches per sample: {n_patches}")

# ── 11. save arrays ───────────────────────────────────────────────────────────
for tag, arr in [("X_train", X_tr),  ("Y_train", Y_tr_seq),
                 ("X_val",   X_va),  ("Y_val",   Y_va_seq),
                 ("X_test",  X_te),  ("Y_test",  Y_te_seq),
                 ("X_all",   X_all), ("Y_all",   Y_all_seq)]:
    np.save(os.path.join(OUT_DIR, f"{tag}.npy"), arr)

with open(os.path.join(OUT_DIR, "scaler_X.pkl"),  "wb") as f: pickle.dump(scaler_X,  f)
with open(os.path.join(OUT_DIR, "scaler_y1.pkl"), "wb") as f: pickle.dump(scaler_y1, f)
np.save(os.path.join(OUT_DIR, "QM_TRAIN_MAX.npy"), np.array([QM_TRAIN_MAX]))

# ── 12. save reference CSVs (times + QM observed + Q TOPKAPI) ────────────────
# test set reference (offset by SEQ_LEN because sequences start after window)
test_ref = pd.DataFrame({
    "Time" : df_test["Time"].iloc[SEQ_LEN:].values,
    "QM"   : df_test["QM"].iloc[SEQ_LEN:].values,
    "Q"    : Q_test["Q"].iloc[SEQ_LEN:].values,
})
test_ref.to_csv(os.path.join(OUT_DIR, "test_reference.csv"), index=False)

# whole-dataset reference (offset by SEQ_LEN)
all_ref = pd.DataFrame({
    "Time" : df["Time"].iloc[SEQ_LEN:].values,
    "QM"   : df["QM"].iloc[SEQ_LEN:].values,
    "Q"    : df_Q["Q"].iloc[SEQ_LEN:].values,
})
all_ref.to_csv(os.path.join(OUT_DIR, "all_reference.csv"), index=False)

print(f"\n✓ test_reference.csv  : {len(test_ref)} rows")
print(f"✓ all_reference.csv   : {len(all_ref)} rows")
print(f"✓ All arrays saved to {OUT_DIR}")