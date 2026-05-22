#!/usr/bin/env python3
"""
dvfs_server.py — long-running inference server for DVFSPredictive.

Spawned once by DVFSPredictive.cc via pipes.  Each DVFS epoch the C++ side
writes one CSV line to stdin; this script responds with one line of
space-separated frequency recommendations (MHz) to stdout.

Request line format (comma-separated):
  nrows, ncols, thermal_limit, horizon, min_mhz, max_mhz, step_mhz,
  [per core: temperature, power, frequency_ghz, utilization,
             cpi_total, cpi_base, cpi_mem_dram, cpi_branch,
             cpi_ifetch, cpi_mem_l1d, ips, active,
             peak_temperature, temp_gradient]

Response line format:
  f0_mhz f1_mhz f2_mhz ...   (one per core, space-separated)
"""

import sys
import os
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from inference import Predictor, WINDOW_SIZE, WINDOW_FEATURES, DELTA_FEATURES, NEIGHBOUR_FEATURES, LABEL_NAMES

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
        return {'neighbour_temp_max': temperatures[core_id],
                'neighbour_temp_mean': temperatures[core_id],
                'neighbour_power_sum': powers[core_id]}
    return {
        'neighbour_temp_max':  max(nb_temps),
        'neighbour_temp_mean': sum(nb_temps) / len(nb_temps),
        'neighbour_power_sum': sum(nb_powers),
    }


def make_candidates(min_mhz, max_mhz, step_mhz):
    freqs = []
    f = max_mhz
    while f >= min_mhz:
        freqs.append(f)
        f -= step_mhz
    return freqs  # descending, highest first


def main():
    predictor     = Predictor()
    windows       = {}   # core_id -> deque of state dicts, newest first
    prev_state    = {}   # core_id -> previous state dict (for delta computation)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        fields = [float(x) for x in line.split(',')]

        nrows         = int(fields[0])
        ncols         = int(fields[1])
        thermal_limit = fields[2]
        horizon       = int(fields[3])
        min_mhz       = int(fields[4])
        max_mhz       = int(fields[5])
        step_mhz      = int(fields[6])

        ncores = nrows * ncols
        horizon_label = 'delta_temp_h{}'.format(horizon)
        if horizon_label not in LABEL_NAMES:
            horizon_label = 'delta_temp_h1'

        # Parse per-core state
        states = []
        for c in range(ncores):
            base = HEADER_FIELDS + c * FIELDS_PER_CORE
            s = dict(zip(WINDOW_FEATURES, fields[base:base + FIELDS_PER_CORE]))
            states.append(s)

        # Compute delta features (requires previous state)
        temperatures = [s['temperature'] for s in states]
        powers       = [s['power']       for s in states]

        for c, s in enumerate(states):
            prev = prev_state.get(c)
            s['delta_temperature'] = s['temperature'] - prev['temperature'] if prev else 0.0
            s['delta_power']       = s['power']       - prev['power']       if prev else 0.0
            s['delta_ips']         = s['ips']         - prev['ips']         if prev else 0.0
            prev_state[c] = s

        # Update sliding windows (newest first, maxlen=WINDOW_SIZE)
        for c in range(ncores):
            if c not in windows:
                windows[c] = deque(maxlen=WINDOW_SIZE)
            windows[c].appendleft(states[c])

        # Choose frequency for each core
        candidates_ghz = [f / 1000.0 for f in make_candidates(min_mhz, max_mhz, step_mhz)]
        result_mhz     = []

        for c in range(ncores):
            window = list(windows[c])

            if len(window) < WINDOW_SIZE:
                # Window not yet full — run at max frequency
                result_mhz.append(max_mhz)
                continue

            neighbours = get_neighbours(c, nrows, ncols, temperatures, powers)
            current_temp = window[0]['temperature']
            chosen_mhz   = min_mhz  # safe fallback

            for freq_ghz in candidates_ghz:
                # Override frequency in lag-0 slot to evaluate this candidate
                candidate_window      = [dict(window[0])] + window[1:]
                candidate_window[0]['frequency_ghz'] = freq_ghz

                try:
                    preds = predictor.predict(candidate_window, neighbours)
                    predicted_temp = current_temp + preds[horizon_label]
                    if predicted_temp < thermal_limit:
                        chosen_mhz = int(freq_ghz * 1000)
                        break   # candidates are descending; take the highest safe one
                except Exception:
                    chosen_mhz = min_mhz
                    break

            result_mhz.append(chosen_mhz)

        sys.stdout.write(' '.join(str(f) for f in result_mhz) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
