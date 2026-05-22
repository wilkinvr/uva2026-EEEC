#!/usr/bin/env python3
"""
Preprocessing script for proactive DVFS predictive model training data.

Loads all maxFreq and ondemand simulation results, parses the Periodic logs,
and builds a per-core per-timestep pandas DataFrame with all features from
PREDICTIVE_MODEL.md.

Usage:
    python preprocessing.py              # prints summary and saves dataset.parquet
    from preprocessing import load_all   # import for use in other scripts
"""

import gzip
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Matches result directory names, e.g.:
#   results_2026-05-18_15.36_SOTA_SingleProgram+4.0GHz+ondemand+fastDVFS_parsec-blackscholes-simsmall-3
NAME_REGEX = re.compile(
    r'results_(\d+-\d+-\d+_\d+\.\d+)_([a-zA-Z0-9_\.\+]+)_((splash2|parsec)-.*)'
)

SCHEDULERS = ('maxFreq', 'ondemand')

# CPI stack metrics to extract (see PREDICTIVE_MODEL.md)
CPI_METRICS = ('total', 'base', 'mem-dram', 'branch', 'ifetch', 'mem-l1d')


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def find_results(results_dir: Path) -> list:
    dirs = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = NAME_REGEX.match(d.name)
        if not m:
            continue
        config = m.group(2)
        if any(s in config for s in SCHEDULERS):
            dirs.append(d)
    return dirs


def _get_scheduler(config: str) -> str:
    for s in SCHEDULERS:
        if s in config:
            return s
    return 'unknown'


# ---------------------------------------------------------------------------
# Log file helpers
# ---------------------------------------------------------------------------

def _open_log(path: Path):
    if path.suffix == '.gz':
        return gzip.open(path, 'rt')
    return open(path, 'r')


def _find_log(result_dir: Path, name: str) -> Path:
    for candidate in (result_dir / name, result_dir / (name + '.gz')):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'{name} not found in {result_dir}')


def _ncores(result_dir: Path) -> int:
    path = _find_log(result_dir, 'PeriodicFrequency.log')
    with _open_log(path) as f:
        header = f.readline().strip().split('\t')
    return len(header)


# ---------------------------------------------------------------------------
# Per-log parsers
# ---------------------------------------------------------------------------

def _parse_component_log(result_dir: Path, name: str, ncores: int,
                          reducer) -> np.ndarray:
    """
    Parse a component-level log (Power or Thermal) into (T, ncores).

    reducer is applied across the C_{i}_* columns for each core:
      - np.sum  for power
      - np.max  for temperature (hotspot)
    """
    path = _find_log(result_dir, name)
    with _open_log(path) as f:
        df = pd.read_csv(f, sep='\t')
    out = np.zeros((len(df), ncores))
    for i in range(ncores):
        cols = [c for c in df.columns if c.startswith(f'C_{i}_')]
        out[:, i] = reducer(df[cols].values, axis=1)
    return out


def parse_power(result_dir: Path, ncores: int) -> np.ndarray:
    """(T, ncores): total power per core [W], matching getPowerOfCore()."""
    return _parse_component_log(result_dir, 'PeriodicPower.log', ncores, np.sum)


def parse_temperature(result_dir: Path, ncores: int) -> np.ndarray:
    """(T, ncores): hotspot temperature per core [°C], matching getTemperatureOfCore()."""
    return _parse_component_log(result_dir, 'PeriodicThermal.log', ncores, np.max)


def parse_frequency(result_dir: Path, ncores: int) -> np.ndarray:
    """(T, ncores): frequency per core [GHz]."""
    path = _find_log(result_dir, 'PeriodicFrequency.log')
    with _open_log(path) as f:
        df = pd.read_csv(f, sep='\t')
    return df[[f'Core{i}' for i in range(ncores)]].values.astype(float)


def parse_cpi_stack(result_dir: Path, ncores: int) -> dict:
    """
    Parse PeriodicCPIStack into a dict mapping metric -> (T, ncores) array.

    The log has no block separators: each timestep is a contiguous block of
    N_METRICS rows. '-' values (zero or inapplicable) are treated as 0.0,
    consistent with getCPIStackPartOfCore().
    """
    path = _find_log(result_dir, 'PeriodicCPIStack.log')
    with _open_log(path) as f:
        lines = f.readlines()

    # Infer block size: count rows until 'total' repeats
    metric_order = []
    for line in lines[1:]:
        metric = line.split('\t')[0].strip()
        if metric == 'total' and metric_order:
            break
        metric_order.append(metric)
    n_metrics = len(metric_order)

    data_lines = lines[1:]
    n_timesteps = len(data_lines) // n_metrics

    result = {m: np.zeros((n_timesteps, ncores)) for m in CPI_METRICS}

    for t in range(n_timesteps):
        block = data_lines[t * n_metrics: (t + 1) * n_metrics]
        for line in block:
            parts = line.strip().split('\t')
            metric = parts[0]
            if metric not in CPI_METRICS:
                continue
            for i in range(ncores):
                raw = parts[i + 1] if i + 1 < len(parts) else '-'
                result[metric][t, i] = 0.0 if raw.strip() == '-' else float(raw)

    return result


# ---------------------------------------------------------------------------
# Per-result DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(result_dir: Path) -> Optional[pd.DataFrame]:
    m = NAME_REGEX.match(result_dir.name)
    if not m:
        return None

    config = m.group(2)
    benchmark = m.group(3)
    scheduler = _get_scheduler(config)

    ncores = _ncores(result_dir)

    power       = parse_power(result_dir, ncores)        # (T, ncores) W
    temperature = parse_temperature(result_dir, ncores)  # (T, ncores) °C
    frequency   = parse_frequency(result_dir, ncores)    # (T, ncores) GHz
    cpi         = parse_cpi_stack(result_dir, ncores)    # dict of (T, ncores)

    T = power.shape[0]

    # Derived per-core features
    cpi_total = cpi['total']
    cpi_base  = cpi['base']

    # Guard against division by zero (inactive cores have cpi_total ~100000,
    # but also guard against any edge-case zero values)
    safe_total = np.where(cpi_total > 0, cpi_total, np.nan)

    utilization = cpi_base / safe_total

    # IPS: matches getIPSOfCore() = 1e6 * freq_MHz / CPI
    # PeriodicFrequency stores GHz, so multiply by 1e9 to get Hz
    ips = (frequency * 1e9) / safe_total

    # System-wide per-timestep features (broadcast to per-core rows)
    peak_temperature = temperature.max(axis=1)              # (T,)
    temp_gradient    = temperature.max(axis=1) - temperature.min(axis=1)  # (T,)

    # Delta features: NaN at t=0
    def delta(arr: np.ndarray) -> np.ndarray:
        d = np.full_like(arr, np.nan)
        d[1:] = arr[1:] - arr[:-1]
        return d

    delta_temperature = delta(temperature)  # (T, ncores)
    delta_power       = delta(power)        # (T, ncores)
    delta_ips         = delta(ips)          # (T, ncores)

    # Build flat records
    timesteps = np.repeat(np.arange(T), ncores)
    core_ids  = np.tile(np.arange(ncores), T)
    t_idx     = timesteps
    c_idx     = core_ids

    df = pd.DataFrame({
        'scheduler':        scheduler,
        'benchmark':        benchmark,
        'result_dir':       result_dir.name,
        'timestep':         t_idx,
        'core_id':          c_idx,
        # Thermal state
        'temperature':      temperature[t_idx, c_idx],
        'peak_temperature': peak_temperature[t_idx],
        'temp_gradient':    temp_gradient[t_idx],
        # Power
        'power':            power[t_idx, c_idx],
        # Control state
        'frequency_ghz':    frequency[t_idx, c_idx],
        'active':           np.where(
                                np.isnan(utilization[t_idx, c_idx]),
                                False,
                                utilization[t_idx, c_idx] > 0.0
                            ).astype(bool),
        # Workload character
        'utilization':      utilization[t_idx, c_idx],
        'cpi_total':        cpi_total[t_idx, c_idx],
        'cpi_base':         cpi_base[t_idx, c_idx],
        'cpi_mem_dram':     cpi['mem-dram'][t_idx, c_idx],
        'cpi_branch':       cpi['branch'][t_idx, c_idx],
        'cpi_ifetch':       cpi['ifetch'][t_idx, c_idx],
        'cpi_mem_l1d':      cpi['mem-l1d'][t_idx, c_idx],
        'ips':              ips[t_idx, c_idx],
        # Delta features
        'delta_temperature': delta_temperature[t_idx, c_idx],
        'delta_power':       delta_power[t_idx, c_idx],
        'delta_ips':         delta_ips[t_idx, c_idx],
    })

    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_all(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    result_dirs = find_results(results_dir)
    if not result_dirs:
        raise RuntimeError(f'No maxFreq or ondemand results found in {results_dir}')

    frames = []
    for result_dir in result_dirs:
        print(f'  Processing {result_dir.name}')
        try:
            df = build_dataframe(result_dir)
            if df is not None:
                frames.append(df)
        except Exception as e:
            print(f'  Warning: skipped {result_dir.name}: {e}', file=sys.stderr)

    if not frames:
        raise RuntimeError('No data could be parsed from any result directory')

    combined = pd.concat(frames, ignore_index=True)
    print(f'Loaded {len(combined):,} rows from {len(frames)} result directories')
    return combined


if __name__ == '__main__':
    print('Scanning results...')
    df = load_all()

    print('\nDataFrame info:')
    print(df.dtypes)
    print(f'\nShape: {df.shape}')
    print(f'\nSchedulers: {df["scheduler"].unique()}')
    print(f'Benchmarks: {df["benchmark"].unique()}')
    print(f'Cores per result: {df["core_id"].max() + 1}')
    print(f'Timesteps (first result): {df[df["result_dir"] == df["result_dir"].iloc[0]]["timestep"].max() + 1}')
    print(f'\nSample rows:')
    print(df.head())

    try:
        out_path = Path(__file__).parent / 'dataset.parquet'
        df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = Path(__file__).parent / 'dataset.csv.gz'
        df.to_csv(out_path, index=False, compression='gzip')
    print(f'\nSaved to {out_path}')
