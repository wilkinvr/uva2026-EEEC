#!/usr/bin/env python3
"""
Training script for the proactive DVFS thermal forecast model.

Trains a multi-output gradient boosted tree model that predicts:
  - ΔT at horizons h=1,2,3  (cumulative temperature change from t)
  - ΔIPS at horizon h=1      (IPS change from t)

Predicting multiple steps ahead makes DVFS proactive: the governor can
evaluate candidate frequencies against the thermal state several epochs
out, not just one step away. The IPS output lets the governor find the
Pareto-optimal frequency — highest frequency that keeps predicted
temperature safe while improving throughput.

See PREDICTIVE_MODEL.md for the full design rationale.

Usage:
    python model_training.py               # trains and saves model
    python model_training.py --no-save     # trains without saving artefacts
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import joblib

PREDICTION_DIR = Path(__file__).parent
RESULTS_DIR    = PREDICTION_DIR.parent / 'results'

# ------------------------------------------------------------------
# Feature configuration
# ------------------------------------------------------------------

WINDOW_SIZE = 5  # number of past timesteps included as lagged features
HORIZON     = 3  # number of future timesteps to predict temperature change for

# Instantaneous features used at each timestep in the window
WINDOW_FEATURES = [
    'temperature',
    'power',
    'frequency_ghz',
    'utilization',
    'cpi_total',
    'cpi_base',
    'cpi_mem_dram',
    'cpi_branch',
    'cpi_ifetch',
    'cpi_mem_l1d',
    'ips',
    'active',
    'peak_temperature',
    'temp_gradient',
]

# Single-timestep (non-lagged) features appended once
DELTA_FEATURES = [
    'delta_temperature',
    'delta_power',
    'delta_ips',
]

# Neighbour core features (current timestep only — not windowed)
# Aggregated so feature count is independent of grid shape.
NEIGHBOUR_FEATURES = [
    'neighbour_temp_max',   # hottest adjacent core
    'neighbour_temp_mean',  # average adjacent core temperature
    'neighbour_power_sum',  # total power of adjacent cores
]

# Model outputs — one per predicted quantity
LABEL_NAMES = (
    [f'delta_temp_h{h}' for h in range(1, HORIZON + 1)]  # cumulative ΔT from t
    + ['delta_ips_h1']                                     # ΔIPS at t+1
)

# Held-out benchmark for the test split (generalisation evaluation)
TEST_BENCHMARK_PATTERN = 'streamcluster'


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    for candidate in (
        PREDICTION_DIR / 'dataset.parquet',
        PREDICTION_DIR / 'dataset.csv.gz',
        PREDICTION_DIR / 'dataset.csv',
    ):
        if candidate.exists():
            print(f'Loading dataset from {candidate}')
            if candidate.suffix == '.parquet':
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate, compression='infer')

    # Dataset not yet built — run preprocessing on the fly
    print('Dataset not found, running preprocessing...')
    sys.path.insert(0, str(PREDICTION_DIR))
    from preprocessing import load_all
    df = load_all()
    return df


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def _make_lag_column_names() -> list:
    names = []
    for lag in range(WINDOW_SIZE):
        suffix = f'_lag{lag}'
        for f in WINDOW_FEATURES:
            names.append(f + suffix)
    return names + DELTA_FEATURES


def _all_feature_names() -> list:
    return _make_lag_column_names() + NEIGHBOUR_FEATURES


def get_grid_neighbours(core_id: int, ncores: int) -> list:
    """
    Return the IDs of orthogonally adjacent cores in the rectangular grid
    used by SchedulerOpen (coreRows × coreColumns, row-major ordering).
    """
    nrows = int(ncores ** 0.5)
    while ncores % nrows != 0:
        nrows -= 1
    ncols = ncores // nrows

    row, col = divmod(core_id, ncols)
    neighbours = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < nrows and 0 <= c < ncols:
            neighbours.append(r * ncols + c)
    return neighbours


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (result_dir, core_id) group sorted by timestep, construct:
      - WINDOW_SIZE lagged copies of WINDOW_FEATURES (lag0 = current, lag1 = t-1, ...)
      - DELTA_FEATURES at the current timestep
      - NEIGHBOUR_FEATURES: aggregated temperature and power of adjacent cores
      - Labels (one column per LABEL_NAME):
          delta_temp_h1..hH  cumulative ΔT = temp[t+h] − temp[t]
          delta_ips_h1        ΔIPS = ips[t+1] − ips[t]

    Cumulative ΔT labels let the governor directly ask "will temperature be
    safe H steps from now?" without chaining per-step predictions.

    Rows where t + HORIZON >= n (insufficient future data) are dropped.
    """
    records = []
    lag_col_names = _make_lag_column_names()

    for result_dir, run_df in df.groupby('result_dir', sort=False):
        ncores = run_df['core_id'].nunique()

        # Pre-sort each core's time series into a dict for fast lookup
        core_data: dict = {}
        for core_id, core_group in run_df.groupby('core_id', sort=False):
            core_data[core_id] = core_group.sort_values('timestep').reset_index(drop=True)

        for core_id in range(ncores):
            group = core_data[core_id]
            n = len(group)

            if n < WINDOW_SIZE + HORIZON:
                continue

            neighbours = get_grid_neighbours(core_id, ncores)

            feat_vals  = group[WINDOW_FEATURES].values.astype(float)
            delta_vals = group[DELTA_FEATURES].values.astype(float)
            temp_vals  = group['temperature'].values.astype(float)
            ips_vals   = group['ips'].values.astype(float)
            meta_cols  = group[['scheduler', 'benchmark', 'result_dir', 'timestep', 'core_id']]

            # Neighbour temperature and power arrays: shape (n, n_neighbours)
            nb_temps  = np.stack(
                [core_data[nb]['temperature'].values.astype(float) for nb in neighbours],
                axis=1,
            )
            nb_powers = np.stack(
                [core_data[nb]['power'].values.astype(float) for nb in neighbours],
                axis=1,
            )

            for t in range(WINDOW_SIZE - 1, n - HORIZON):
                # lag0 = current (t), lag1 = t-1, ..., lag(k-1) = t-(k-1)
                window = feat_vals[t - WINDOW_SIZE + 1: t + 1][::-1].flatten()
                deltas = delta_vals[t]

                row = dict(zip(lag_col_names, np.concatenate([window, deltas])))
                row['neighbour_temp_max']  = nb_temps[t].max()
                row['neighbour_temp_mean'] = nb_temps[t].mean()
                row['neighbour_power_sum'] = nb_powers[t].sum()

                # Multi-step temperature labels: cumulative ΔT from current timestep
                for h in range(1, HORIZON + 1):
                    row[f'delta_temp_h{h}'] = temp_vals[t + h] - temp_vals[t]

                # IPS label: single-step change
                row['delta_ips_h1'] = ips_vals[t + 1] - ips_vals[t]

                row.update(meta_cols.iloc[t].to_dict())
                records.append(row)

    result = pd.DataFrame(records)
    result[_all_feature_names()] = result[_all_feature_names()].astype(float)
    result[LABEL_NAMES] = result[LABEL_NAMES].astype(float)
    return result


# ------------------------------------------------------------------
# Train / validation / test split
# ------------------------------------------------------------------

def split_dataset(windowed: pd.DataFrame):
    """
    Split by result_dir (not by timestep) to avoid temporal leakage.
    Test set: any run whose benchmark contains TEST_BENCHMARK_PATTERN.
    Remaining runs: 80 % train, 20 % validation.
    """
    runs = windowed['result_dir'].unique()

    test_runs  = [r for r in runs if TEST_BENCHMARK_PATTERN in r]
    other_runs = [r for r in runs if r not in test_runs]

    rng = np.random.default_rng(seed=42)
    rng.shuffle(other_runs)
    n_val    = max(1, int(len(other_runs) * 0.2))
    val_runs = other_runs[:n_val]
    train_runs = other_runs[n_val:]

    train = windowed[windowed['result_dir'].isin(train_runs)]
    val   = windowed[windowed['result_dir'].isin(val_runs)]
    test  = windowed[windowed['result_dir'].isin(test_runs)]

    print(f'\nSplit (by result_dir):')
    print(f'  train : {len(train_runs)} runs, {len(train):,} rows')
    print(f'  val   : {len(val_runs)} runs, {len(val):,} rows')
    print(f'  test  : {len(test_runs)} runs, {len(test):,} rows  '
          f'[benchmark: {TEST_BENCHMARK_PATTERN}]')

    if not train_runs:
        raise RuntimeError('Training split is empty — need more result directories.')

    return train, val, test


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def build_model():
    if HAS_XGBOOST:
        print(f'\nUsing MultiOutputRegressor(XGBRegressor) — {len(LABEL_NAMES)} outputs')
        base = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        print(f'\nUsing MultiOutputRegressor(GradientBoostingRegressor) — {len(LABEL_NAMES)} outputs')
        base = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    return MultiOutputRegressor(base, n_jobs=-1)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, scaler: StandardScaler, split: pd.DataFrame,
             feature_cols: list) -> dict:
    """Returns per-label MAE, RMSE, and max-error."""
    if split.empty:
        return {}
    X      = scaler.transform(split[feature_cols].values)
    Y      = split[LABEL_NAMES].values          # (n, n_outputs)
    Y_pred = np.array(model.predict(X))         # (n, n_outputs)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred[:, np.newaxis]

    results = {'n': len(Y)}
    for i, label in enumerate(LABEL_NAMES):
        err = np.abs(Y[:, i] - Y_pred[:, i])
        results[label] = {
            'mae':       float(mean_absolute_error(Y[:, i], Y_pred[:, i])),
            'rmse':      float(root_mean_squared_error(Y[:, i], Y_pred[:, i])),
            'max_error': float(err.max()),
        }
    return results


def print_metrics(split_name: str, metrics: dict) -> None:
    if not metrics:
        print(f'  {split_name}: (no data)')
        return
    unit = {'delta_ips_h1': 'IPS'}
    print(f'  {split_name}  (n={metrics["n"]:,}):')
    for label in LABEL_NAMES:
        m = metrics[label]
        u = unit.get(label, '°C')
        print(f'    {label:20s}  MAE={m["mae"]:9.3f} {u}  '
              f'RMSE={m["rmse"]:9.3f} {u}  '
              f'max={m["max_error"]:9.3f} {u}')


# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

def print_top_features(model, feature_cols: list, n: int = 15) -> None:
    # MultiOutputRegressor: average importances across per-output sub-models
    if hasattr(model, 'estimators_'):
        all_imp = [
            est.feature_importances_
            for est in model.estimators_
            if hasattr(est, 'feature_importances_')
        ]
        if not all_imp:
            return
        importances = np.mean(all_imp, axis=0)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    idx = np.argsort(importances)[::-1][:n]
    print(f'\nTop {n} features by mean importance across outputs:')
    for rank, i in enumerate(idx, 1):
        print(f'  {rank:2d}. {feature_cols[i]:40s} {importances[i]:.4f}')


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(save: bool = True) -> None:
    # Load and prepare data
    raw = load_dataset()
    print(f'Raw dataset: {len(raw):,} rows, {raw["result_dir"].nunique()} runs')

    print('\nBuilding sliding-window features (window={})...'.format(WINDOW_SIZE))
    windowed = build_windows(raw)
    print(f'Windowed dataset: {len(windowed):,} rows')

    feature_cols = _all_feature_names()

    # Split
    train, val, test = split_dataset(windowed)

    # Scale — fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols].values)

    # Train
    model = build_model()
    print(f'Training...')
    Y_train = train[LABEL_NAMES].values
    model.fit(X_train, Y_train)

    # Evaluate
    print(f'\nMetrics — outputs: {LABEL_NAMES}')
    print_metrics('train', evaluate(model, scaler, train, feature_cols))
    print_metrics('val',   evaluate(model, scaler, val,   feature_cols))
    print_metrics('test',  evaluate(model, scaler, test,  feature_cols))

    print_top_features(model, feature_cols)

    # Save artefacts
    if save:
        import json

        # Save scaler as plain JSON so inference.py needs no sklearn at runtime
        scaler_path = PREDICTION_DIR / 'scaler.json'
        with open(scaler_path, 'w') as fh:
            json.dump({'mean_': scaler.mean_.tolist(),
                       'scale_': scaler.scale_.tolist()}, fh)
        print(f'\nSaved scaler → {scaler_path}')

        if HAS_XGBOOST and hasattr(model, 'estimators_'):
            # Save each XGBoost sub-model in its native JSON format.
            # XGBoost JSON models are stable across library versions,
            # so the container can load them with any xgboost >= 1.0.
            for i, est in enumerate(model.estimators_):
                p = PREDICTION_DIR / f'model_{i}.json'
                est.save_model(str(p))
                print(f'Saved model  → {p}')
            meta_path = PREDICTION_DIR / 'model_meta.json'
            with open(meta_path, 'w') as fh:
                json.dump({'labels': LABEL_NAMES}, fh)
            print(f'Saved meta   → {meta_path}')
        else:
            # Fallback: joblib for non-XGBoost models
            model_path = PREDICTION_DIR / 'model.joblib'
            joblib.dump(model, model_path)
            print(f'Saved model  → {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help='Skip saving model artefacts')
    args = parser.parse_args()
    main(save=args.save)
