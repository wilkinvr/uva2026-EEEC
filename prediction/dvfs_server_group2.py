#!/usr/bin/env python3
"""
dvfs_server_group2.py — inference server for DVFSGroup2.

Spawned once by DVFSGroup2.cc via pipes.  Each DVFS epoch the C++ side
writes one CSV line to stdin; this script responds with one line of
space-separated delta_temp_hN floats (one per core) to stdout, where N is
the horizon value from the request header.

The C++ side applies dvfs_thermal's band/bisect logic using the predicted
effective temperature (current_temp + delta_temp_hN) instead of the measured
temperature.  The frequency_ghz field in the request carries the candidate
frequency the C++ side is about to apply, so the prediction is made at that
frequency rather than the previous one.

Request line format: identical to dvfs_server.py.
  nrows, ncols, thermal_limit, horizon, min_mhz, max_mhz, step_mhz,
  [per core: temperature, power, frequency_ghz, utilization,
             cpi_total, cpi_base, cpi_mem_dram, cpi_branch,
             cpi_ifetch, cpi_mem_l1d, ips, active,
             peak_temperature, temp_gradient]

Response line format:
  dt0 dt1 dt2 ...   (delta_temp_h1 per core, space-separated floats)
  During window fill-up, returns '0.0' for each core.
"""

import sys
import os
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from inference import Predictor, WINDOW_SIZE, WINDOW_FEATURES, LABEL_NAMES

FIELDS_PER_CORE = len(WINDOW_FEATURES)   # 14
HEADER_FIELDS   = 7


def get_neighbours(core_id, nrows, ncols, temperatures, powers):
    row, col = divmod(core_id, ncols)
    nb_temps, nb_powers = [], []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < nrows and 0 <= c < ncols:
            nb = r * ncols + c
            nb_temps.append(temperatures[nb])
            nb_powers.append(powers[nb])
    if not nb_temps:
        return {'neighbour_temp_max':  temperatures[core_id],
                'neighbour_temp_mean': temperatures[core_id],
                'neighbour_power_sum': powers[core_id]}
    return {
        'neighbour_temp_max':  max(nb_temps),
        'neighbour_temp_mean': sum(nb_temps) / len(nb_temps),
        'neighbour_power_sum': sum(nb_powers),
    }


def main():
    predictor  = Predictor()
    windows    = {}   # core_id -> deque of state dicts, newest first
    prev_state = {}   # core_id -> previous state dict (for delta computation)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        fields = [float(x) for x in line.split(',')]

        nrows   = int(fields[0])
        ncols   = int(fields[1])
        horizon = int(fields[3])

        horizon_label = 'delta_temp_h{}'.format(horizon)
        if horizon_label not in LABEL_NAMES:
            horizon_label = 'delta_temp_h1'

        ncores = nrows * ncols

        # Parse per-core state
        states = []
        for c in range(ncores):
            base = HEADER_FIELDS + c * FIELDS_PER_CORE
            s = dict(zip(WINDOW_FEATURES, fields[base:base + FIELDS_PER_CORE]))
            states.append(s)

        temperatures = [s['temperature'] for s in states]
        powers       = [s['power']       for s in states]

        # Compute delta features
        for c, s in enumerate(states):
            prev = prev_state.get(c)
            s['delta_temperature'] = s['temperature'] - prev['temperature'] if prev else 0.0
            s['delta_power']       = s['power']       - prev['power']       if prev else 0.0
            s['delta_ips']         = s['ips']         - prev['ips']         if prev else 0.0
            prev_state[c] = s

        # Update sliding windows
        for c in range(ncores):
            if c not in windows:
                windows[c] = deque(maxlen=WINDOW_SIZE)
            windows[c].appendleft(states[c])

        # Predict delta_temp_hN at the candidate frequency for each core
        result = []
        for c in range(ncores):
            window = list(windows[c])

            if len(window) < WINDOW_SIZE:
                result.append(0.0)
                continue

            neighbours = get_neighbours(c, nrows, ncols, temperatures, powers)
            try:
                preds = predictor.predict(window, neighbours)
                result.append(preds[horizon_label])
            except Exception:
                result.append(0.0)

        sys.stdout.write(' '.join('{:.4f}'.format(dt) for dt in result) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
