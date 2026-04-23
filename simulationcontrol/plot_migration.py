"""
Assignment 2 - Thread Migration: Parse results and generate comparison charts.
Run from inside the simulationcontrol directory:
    python3 plot_migration.py
"""

import os
import gzip
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

# Map config tag -> readable label and x-axis order
MIGRATION_CONFIGS = [
    ('migrationOff',                             'No Migration\n(baseline)'),
    ('migrationHeatAndRun+migrationEpochSlow',   'Heat&Run\nSlow\n(epoch=1ms)'),
    ('migrationHeatAndRun+migrationEpochMedium', 'Heat&Run\nMedium\n(epoch=0.1ms)'),
    ('migrationHeatAndRun+migrationEpochFast',   'Heat&Run\nFast\n(epoch=0.01ms)'),
]

# Number of floorplan components per core in gainestown_4_core layout
# Header: L3, then C_0_* (20 cols), C_1_* (20 cols), C_2_* (20 cols), C_3_* (20 cols)
N_PREFIX_COLS = 1   # L3
N_COLS_PER_CORE = 21  # C_N_IALU .. C_N_FPRF


def find_result(benchmark, config_tag):
    """Return the most recent result directory for the given benchmark and config."""
    matches = []
    for d in os.listdir(RESULTS_DIR):
        m = re.match(r'results_(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2})_(.+)_(parsec-.+)', d)
        if not m:
            continue
        config_part = m.group(2)
        bench_part  = m.group(3)
        if not bench_part.startswith(benchmark):
            continue
        if config_part == '4.0GHz+maxFreq+slowDVFS+' + config_tag:
            matches.append((m.group(1), d))   # (timestamp, dirname)
    if not matches:
        return None
    return os.path.join(RESULTS_DIR, sorted(matches)[-1][1])


def get_response_time(result_dir):
    """Simulated application time in ms from sim.out."""
    sim_out = os.path.join(result_dir, 'sim.out')
    if not os.path.exists(sim_out):
        return None
    with open(sim_out) as f:
        content = f.read()
    m = re.search(r'^\s*Time \(ns\)\s*\|\s*([\d.]+)', content, re.MULTILINE)
    return float(m.group(1)) / 1e6 if m else None


def _read_thermal_gz(result_dir):
    """Return list of rows (each a list of floats), skipping the header."""
    path = os.path.join(result_dir, 'PeriodicThermal.log.gz')
    if not os.path.exists(path):
        return None
    rows = []
    with gzip.open(path, 'rt') as f:
        first = True
        for line in f:
            if first:               # skip header
                first = False
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append([float(x) for x in stripped.split()])
            except ValueError:
                continue
    return rows if rows else None


def get_peak_temperature(result_dir):
    """Overall peak temperature (°C) across all cores and time steps."""
    rows = _read_thermal_gz(result_dir)
    if rows is None:
        return None
    return max(max(row) for row in rows)


def get_avg_core_peak_temp(result_dir):
    """
    Time-averaged peak temperature of the hottest core (°C).
    For each time step we take the per-core max, then average over time.
    Columns: L3 (col 0), C_0 (cols 1-21), C_1 (cols 22-42), C_2 (cols 43-63), C_3 (cols 64-84)
    """
    rows = _read_thermal_gz(result_dir)
    if rows is None:
        return None
    n_cores = 4
    hottest_per_row = []
    for row in rows:
        core_maxes = []
        for c in range(n_cores):
            start = N_PREFIX_COLS + c * N_COLS_PER_CORE
            end   = start + N_COLS_PER_CORE
            segment = row[start:end]
            if segment:
                core_maxes.append(max(segment))
        if core_maxes:
            hottest_per_row.append(max(core_maxes))
    return float(np.mean(hottest_per_row)) if hottest_per_row else None


def get_per_core_avg_max(result_dir):
    """
    Time-averaged maximum temperature per core (°C).
    Returns list of 4 values.
    """
    rows = _read_thermal_gz(result_dir)
    if rows is None:
        return None
    n_cores = 4
    per_core = [[] for _ in range(n_cores)]
    for row in rows:
        for c in range(n_cores):
            start = N_PREFIX_COLS + c * N_COLS_PER_CORE
            end   = start + N_COLS_PER_CORE
            segment = row[start:end]
            if segment:
                per_core[c].append(max(segment))
    return [float(np.mean(v)) if v else None for v in per_core]


def get_avg_power(result_dir):
    """Average total chip power (W) from PeriodicPower.log.gz."""
    path = os.path.join(result_dir, 'PeriodicPower.log.gz')
    if not os.path.exists(path):
        return None
    totals = []
    with gzip.open(path, 'rt') as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                totals.append(sum(float(x) for x in stripped.split()))
            except ValueError:
                continue
    return float(np.mean(totals)) if totals else None


def parse_all():
    data = {}
    for benchmark in ('parsec-blackscholes', 'parsec-streamcluster'):
        data[benchmark] = {
            'labels': [], 'response_time': [],
            'peak_temp': [], 'avg_core_peak': [],
            'avg_power': [], 'per_core_avg': [],
            'result_dir': [],
        }
        for config_tag, label in MIGRATION_CONFIGS:
            rdir = find_result(benchmark, config_tag)
            if rdir is None:
                print(f'  [MISSING] {benchmark} / {config_tag}')
                data[benchmark]['labels'].append(label)
                for k in ('response_time', 'peak_temp', 'avg_core_peak', 'avg_power'):
                    data[benchmark][k].append(None)
                data[benchmark]['per_core_avg'].append(None)
                data[benchmark]['result_dir'].append(None)
                continue

            rt  = get_response_time(rdir)
            pt  = get_peak_temperature(rdir)
            acp = get_avg_core_peak_temp(rdir)
            ap  = get_avg_power(rdir)
            pca = get_per_core_avg_max(rdir)

            print(f'  {benchmark} / {config_tag}:')
            print(f'    RT={rt:.2f}ms  PeakT={pt:.2f}°C  AvgCoreT={acp:.2f}°C  AvgP={ap:.3f}W')
            print(f'    PerCoreAvgMax: C0={pca[0]:.2f} C1={pca[1]:.2f} C2={pca[2]:.2f} C3={pca[3]:.2f}')

            data[benchmark]['labels'].append(label)
            data[benchmark]['response_time'].append(rt)
            data[benchmark]['peak_temp'].append(pt)
            data[benchmark]['avg_core_peak'].append(acp)
            data[benchmark]['avg_power'].append(ap)
            data[benchmark]['per_core_avg'].append(pca)
            data[benchmark]['result_dir'].append(rdir)
    return data


def _bar_chart(ax, vals, labels, ylabel, title, hatches, colors, zoom=True):
    x = np.arange(len(labels))
    for i, (v, h, c) in enumerate(zip(vals, hatches, colors)):
        if v is not None:
            ax.bar(i, v, color=c, edgecolor='black', hatch=h, width=0.6)
        else:
            ax.bar(i, 0, color='red', width=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    # zoom y-axis to show detail even when values are close
    valid = [v for v in vals if v is not None]
    if zoom and valid:
        lo, hi = min(valid), max(valid)
        margin = max((hi - lo) * 2, hi * 0.02)
        ax.set_ylim(max(0, lo - margin), hi + margin)
    for i, v in enumerate(vals):
        if v is not None:
            ax.text(i, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')


def plot(data):
    hatches = ['', '///', 'xxx', '...']
    colors  = ['white', 'lightgray', 'darkgray', 'dimgray']

    for benchmark in ('parsec-blackscholes', 'parsec-streamcluster'):
        bname = benchmark.replace('parsec-', '').capitalize()
        labels = data[benchmark]['labels']

        # ── Figure 1: 3-metric summary ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Thread Migration Effects — {bname}', fontsize=13, fontweight='bold')

        _bar_chart(axes[0], data[benchmark]['response_time'],  labels, 'Response Time (ms)',     'Response Time',      hatches, colors)
        _bar_chart(axes[1], data[benchmark]['peak_temp'],      labels, 'Peak Temperature (°C)',  'Peak Temperature',   hatches, colors)
        _bar_chart(axes[2], data[benchmark]['avg_power'],      labels, 'Average Power (W)',       'Average Power',      hatches, colors)

        plt.tight_layout()
        out1 = os.path.join(RESULTS_DIR, f'migration_comparison_{benchmark.replace("parsec-","")}.png')
        plt.savefig(out1, dpi=150, bbox_inches='tight')
        print(f'Saved: {out1}')
        plt.close()

        # ── Figure 2: per-core temperature breakdown ─────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.suptitle(f'Per-Core Avg-Max Temperature — {bname}', fontsize=13, fontweight='bold')

        n_configs = len(MIGRATION_CONFIGS)
        n_cores   = 4
        bar_w     = 0.18
        core_colors  = ['white', 'lightgray', 'darkgray', 'dimgray']
        core_hatches = ['', '///', 'xxx', '...']
        core_labels  = [f'Core {i}' for i in range(n_cores)]

        for ci in range(n_cores):
            offsets = np.arange(n_configs) + (ci - 1.5) * bar_w
            vals_c  = []
            for cfg_idx in range(n_configs):
                pca = data[benchmark]['per_core_avg'][cfg_idx]
                vals_c.append(pca[ci] if pca and pca[ci] is not None else 0)
            ax2.bar(offsets, vals_c, width=bar_w, label=core_labels[ci],
                    color=core_colors[ci], hatch=core_hatches[ci], edgecolor='black')

        ax2.set_xticks(np.arange(n_configs))
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylabel('Avg-Max Temperature (°C)')
        ax2.legend(loc='lower right')

        # zoom
        all_vals = [v for pca in data[benchmark]['per_core_avg'] if pca
                    for v in pca if v is not None]
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = max((hi - lo) * 2, hi * 0.02)
            ax2.set_ylim(max(0, lo - margin), hi + margin)

        plt.tight_layout()
        out2 = os.path.join(RESULTS_DIR, f'migration_percore_{benchmark.replace("parsec-","")}.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        print(f'Saved: {out2}')
        plt.close()


if __name__ == '__main__':
    print('Parsing results...')
    data = parse_all()
    print('\nGenerating charts...')
    plot(data)
    print('Done!')
