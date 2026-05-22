#!/usr/bin/env python3
"""
inference.py — single-prediction interface for the proactive DVFS model.

The model predicts, for a given core at the current timestep:
  delta_temp_h1..h3  cumulative temperature change (°C) at t+1, t+2, t+3
  delta_ips_h1       IPS change at t+1

Artefacts produced by model_training.py:
  scaler.json      — StandardScaler mean/scale as plain JSON arrays
  model_0.json ..  — one XGBoost model per output, in XGBoost's native JSON
  model_meta.json  — label order

Only numpy and xgboost are required at runtime (no sklearn, no joblib).

Usage:
    from inference import Predictor
    predictor = Predictor()
    preds = predictor.predict(window, neighbours)
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import xgboost as xgb

PREDICTION_DIR = Path(__file__).parent

# Must stay in sync with model_training.py
WINDOW_SIZE = 5

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

DELTA_FEATURES = [
    'delta_temperature',
    'delta_power',
    'delta_ips',
]

NEIGHBOUR_FEATURES = [
    'neighbour_temp_max',
    'neighbour_temp_mean',
    'neighbour_power_sum',
]

LABEL_NAMES = [
    'delta_temp_h1',
    'delta_temp_h2',
    'delta_temp_h3',
    'delta_ips_h1',
]

N_FEATURES = WINDOW_SIZE * len(WINDOW_FEATURES) + len(DELTA_FEATURES) + len(NEIGHBOUR_FEATURES)


class Predictor:
    def __init__(self, model_dir=None, scaler_path=None):
        if model_dir is None:
            model_dir = PREDICTION_DIR
        if scaler_path is None:
            scaler_path = PREDICTION_DIR / 'scaler.json'

        # Load scaler from plain JSON — no sklearn dependency
        with open(str(scaler_path)) as fh:
            d = json.load(fh)
        self._mean  = np.array(d['mean_'],  dtype=float)
        self._scale = np.array(d['scale_'], dtype=float)

        # Load label order from meta file if present, else use default
        meta_path = Path(str(model_dir)) / 'model_meta.json'
        if meta_path.exists():
            with open(str(meta_path)) as fh:
                self._labels = json.load(fh)['labels']
        else:
            self._labels = LABEL_NAMES

        # Load one XGBoost Booster per output (native API — no sklearn dependency)
        self._models = []
        for i in range(len(self._labels)):
            m = xgb.Booster()
            m.load_model(str(Path(str(model_dir)) / ('model_{}.json'.format(i))))
            self._models.append(m)

    def predict(self, window, neighbours):
        """
        Predict thermal and IPS changes for a single core at the current timestep.

        Parameters
        ----------
        window : list of WINDOW_SIZE dicts, most-recent first (index 0 = current t).
            Each dict must contain all keys in WINDOW_FEATURES and DELTA_FEATURES.
            The DELTA_FEATURES are read from window[0] (current timestep only).

        neighbours : dict with keys from NEIGHBOUR_FEATURES:
            'neighbour_temp_max'   — hottest adjacent core temperature (°C)
            'neighbour_temp_mean'  — mean adjacent core temperature (°C)
            'neighbour_power_sum'  — total power of adjacent cores (W)

        Returns
        -------
        dict mapping each label in LABEL_NAMES to its predicted float value:
            'delta_temp_h1' .. 'delta_temp_h3'  cumulative ΔT from t (°C)
            'delta_ips_h1'                       ΔIPS at t+1
        """
        if len(window) != WINDOW_SIZE:
            raise ValueError(
                'window must have exactly {} entries, got {}'.format(WINDOW_SIZE, len(window))
            )

        # Lag features: lag0 = window[0] (t), lag1 = window[1] (t-1), ...
        lag_values = [
            window[lag][f]
            for lag in range(WINDOW_SIZE)
            for f in WINDOW_FEATURES
        ]

        delta_values     = [window[0][f] for f in DELTA_FEATURES]
        neighbour_values = [neighbours[f] for f in NEIGHBOUR_FEATURES]

        row = np.array(lag_values + delta_values + neighbour_values, dtype=float).reshape(1, -1)
        row_scaled = (row - self._mean) / self._scale
        dm = xgb.DMatrix(row_scaled)
        preds = [float(m.predict(dm)[0]) for m in self._models]

        return dict(zip(self._labels, preds))


class CorePredictor:
    """
    Stateful wrapper that maintains the sliding window for one core.
    Call push() after each DVFS epoch, then predict() to get the forecast.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self._window = deque(maxlen=WINDOW_SIZE)
        self._prev_temperature = None
        self._prev_power       = None
        self._prev_ips         = None

    @property
    def ready(self):
        return len(self._window) == WINDOW_SIZE

    def push(self, state):
        """
        Record the current timestep state.  state must contain all WINDOW_FEATURES.
        delta_temperature / delta_power / delta_ips are computed automatically.
        """
        entry = dict(state)
        entry['delta_temperature'] = (
            state['temperature'] - self._prev_temperature
            if self._prev_temperature is not None else float('nan')
        )
        entry['delta_power'] = (
            state['power'] - self._prev_power
            if self._prev_power is not None else float('nan')
        )
        entry['delta_ips'] = (
            state['ips'] - self._prev_ips
            if self._prev_ips is not None else float('nan')
        )

        self._window.appendleft(entry)   # index 0 = most recent
        self._prev_temperature = state['temperature']
        self._prev_power       = state['power']
        self._prev_ips         = state['ips']

    def predict(self, neighbours, candidate_frequency_ghz=None):
        """
        Return predictions for the current window state.
        If candidate_frequency_ghz is given, the frequency in the lag-0 slot is
        temporarily overridden so the model evaluates that candidate frequency.
        """
        if not self.ready:
            raise RuntimeError(
                'Need {} timesteps before predicting; have {}'.format(WINDOW_SIZE, len(self._window))
            )

        window = list(self._window)   # copy so override doesn't mutate state
        if candidate_frequency_ghz is not None:
            window[0] = dict(window[0])
            window[0]['frequency_ghz'] = candidate_frequency_ghz

        return self.predictor.predict(window, neighbours)

    def best_frequency(self, candidates, neighbours, thermal_limit=80.0):
        """
        Return the highest candidate frequency (GHz) whose predicted t+1
        temperature stays below thermal_limit.  Falls back to the lowest
        candidate if all exceed the limit.
        """
        current_temp = self._window[0]['temperature']
        for freq in sorted(candidates, reverse=True):
            preds = self.predict(neighbours, candidate_frequency_ghz=freq)
            if current_temp + preds['delta_temp_h1'] < thermal_limit:
                return freq
        return min(candidates)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run a single test prediction')
    parser.add_argument('--freq', type=float, default=4.0, help='Frequency (GHz) to evaluate')
    args = parser.parse_args()

    predictor = Predictor()
    print('Model loaded: {} outputs, {} features'.format(len(LABEL_NAMES), N_FEATURES))

    # Synthetic state: warm idle core at 60 °C
    state = {
        'temperature':   60.0,
        'power':          1.5,
        'frequency_ghz':  args.freq,
        'utilization':    0.4,
        'cpi_total':      2.0,
        'cpi_base':       0.8,
        'cpi_mem_dram':   0.3,
        'cpi_branch':     0.1,
        'cpi_ifetch':     0.1,
        'cpi_mem_l1d':    0.2,
        'ips':            2e9,
        'active':         1.0,
        'peak_temperature': 62.0,
        'temp_gradient':    3.0,
        'delta_temperature': 0.1,
        'delta_power':       0.05,
        'delta_ips':         1e7,
    }
    window = [state] * WINDOW_SIZE
    neighbours = {
        'neighbour_temp_max':  61.0,
        'neighbour_temp_mean': 59.5,
        'neighbour_power_sum':  4.0,
    }

    preds = predictor.predict(window, neighbours)
    print('\nPredictions at {} GHz (current temp = {} C):'.format(args.freq, state['temperature']))
    for label, val in preds.items():
        unit = 'C' if 'temp' in label else 'IPS'
        print('  {:<20s}  {:+.3f} {}'.format(label, val, unit))
    print('\n  predicted temp at t+1: {:.1f} C'.format(
        state['temperature'] + preds['delta_temp_h1']
    ))
