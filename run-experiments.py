#!/usr/bin/env python3
"""
run-experiments.py — run experiments in parallel Docker containers with a live tmux view.

Edit get_experiments() to define what to run. Each entry in the returned list
gets its own container and tmux pane.

This script was made with the help of LLMs.

Usage:
  python3 run-experiments.py              # launch experiments
  python3 run-experiments.py --build      # rebuild image, then launch
  python3 run-experiments.py --list       # list experiments without launching
  python3 run-experiments.py --dry-run    # show docker commands without executing
"""

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Experiment helpers (mirrors simulationcontrol/run.py — no host-side deps)
# ---------------------------------------------------------------------------

_THREAD_COUNTS = {
    'parsec-blackscholes':  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'parsec-bodytrack':     [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'parsec-canneal':       [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'parsec-dedup':         [4, 7, 10, 13, 16],
    'parsec-fluidanimate':  [2, 3, 0, 5, 0, 0, 0, 9],
    'parsec-streamcluster': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'parsec-swaptions':     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'parsec-x264':          [1, 3, 4, 5, 6, 7, 8, 9],
    'splash2-barnes':       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-cholesky':     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-fft':          [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
    'splash2-fmm':          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-lu.cont':      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-lu.ncont':     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-ocean.cont':   [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
    'splash2-ocean.ncont':  [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
    'splash2-radiosity':    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-radix':        [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
    'splash2-raytrace':     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-water.nsq':    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'splash2-water.sp':     [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
}


class Infeasible(Exception):
    pass


def get_instance(benchmark, parallelism, input_set='small'):
    ps = _THREAD_COUNTS[benchmark]
    if parallelism <= 0 or parallelism not in ps:
        raise Infeasible(f'{benchmark} does not support parallelism={parallelism}')
    p = ps.index(parallelism) + 1
    if benchmark.startswith('parsec') and not input_set.startswith('sim'):
        input_set = 'sim' + input_set
    return f'{benchmark}-{input_set}-{p}'


def get_workload(benchmark, cores, parallelism=None, number_tasks=None, input_set='small'):
    if parallelism is not None:
        number_tasks = math.floor(cores / parallelism)
        return get_workload(benchmark, cores, number_tasks=number_tasks, input_set=input_set)
    elif number_tasks is not None:
        if number_tasks == 0:
            if cores == 0:
                return []
            raise Infeasible()
        parallelism = math.ceil(cores / number_tasks)
        for p in reversed(range(1, min(cores, parallelism) + 1)):
            try:
                b = get_instance(benchmark, p, input_set=input_set)
                return [b] + get_workload(benchmark, cores - p, number_tasks=number_tasks - 1, input_set=input_set)
            except Infeasible:
                pass    
        raise Infeasible()
    else:
        raise Exception('either parallelism or number_tasks needs to be set')


# ---------------------------------------------------------------------------
# Experiment list — edit this function to define your experiments
#
# Return a list of (label, base_config, benchmark) where:
#   label       — short string used as the tmux pane title
#   base_config — list of config strings, same as the first arg to run()
#   benchmark   — benchmark string or comma-separated multi-program string
# ---------------------------------------------------------------------------


def get_experiments():
    experiments = []

    benchmarks = [
        # ('parsec-blackscholes',  4, 'simdev'),
        # ('parsec-streamcluster', 4, 'simdev'),
        ('parsec-blackscholes',  4, 'simsmall'),
        ('parsec-streamcluster', 4, 'simsmall'),
        ('parsec-swaptions',     4, 'simsmall'),
        ('splash2-fft',          4, 'small'),
        ('splash2-lu.cont',      4, 'small'),
        ('splash2-radix',        4, 'small'),
    ]

    # (label_suffix, cfg_flags)
    # maxFreq: unconstrained throughput ceiling
    # ondemand: standard industry reactive baseline
    # thermal_binary: binary step reactive thermal governor
    # predictive_hN: our proactive ML governor at horizon N
    # group2_hN: ML-predicted temperature fed into thermal band logic
    configs = [
        # ('maxFreq',              ['maxFreq']),
        # ('ondemand',             ['ondemand']),
        # ('thermal',              ['thermal_binary']),
        ('predictive_h1',        ['predictive', 'horizon1']),
        ('predictive_h2',        ['predictive', 'horizon2']),
        ('predictive_h3',        ['predictive', 'horizon3']),
        ('group2_h1',            ['group2', 'horizon1']),
        ('group2_h2',            ['group2', 'horizon2']),
        ('group2_h3',            ['group2', 'horizon3']),
    ]

    # Single-program
    for benchmark_name, parallelism, input_set in benchmarks:
        instance = get_instance(benchmark_name, parallelism, input_set=input_set)
        short = benchmark_name.replace('parsec-', '').replace('splash2-', '')
        for label_suffix, cfg_flags in configs:
            base_config = ['SOTA_SingleProgram', '4.0GHz', *cfg_flags, 'fastDVFS']
            label = f"{short}_{label_suffix}"
            experiments.append((label, base_config, instance))

    # Multi-program — blackscholes + streamcluster (p=2 each, fits in 4 cores)
    multi_instance = ','.join(
        get_instance(bm, 2, input_set='simsmall')
        for bm in ('parsec-blackscholes', 'parsec-streamcluster')
    )
    for label_suffix, cfg_flags in configs:
        base_config = ['SOTA_MultiProgram', '4.0GHz', *cfg_flags, 'fastDVFS']
        label = f"multi_{label_suffix}"
        experiments.append((label, base_config, multi_instance))

    return experiments


def get_training_experiments():
    experiments = []

    benchmarks = [
        ('parsec-blackscholes',  4, 'simsmall'),
        ('parsec-swaptions',     4, 'simsmall'),
        ('parsec-streamcluster', 4, 'simsmall'),
        ('splash2-fft',          4, 'small'),
        ('splash2-lu.cont',      4, 'small'),
        ('splash2-radix',        4, 'small'),
    ]

    frequencies = [f'{f:.1f}GHz' for f in [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]]

    for benchmark_name, parallelism, input_set in benchmarks:
        instance = get_instance(benchmark_name, parallelism, input_set=input_set)
        short = benchmark_name.replace('parsec-', '').replace('splash2-', '')
        for freq in frequencies:
            base_config = ['SOTA_SingleProgram', freq, 'maxFreq', 'fastDVFS']
            label = f"{short}_maxFreq_{freq}"
            experiments.append((label, base_config, instance))

    return experiments


# ---------------------------------------------------------------------------
# Docker + tmux launcher
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

_UBUNTU_VERSION = '18.04'
_USER = os.environ.get('USER', 'user')
DOCKER_EXPERIMENT_IMAGE = f'ubuntu:{_UBUNTU_VERSION}-sniper-experiments-{_USER}'
DEFAULT_RESULTS_DIR = os.path.join(HERE, 'results')
DEFAULT_SESSION = 'experiments'


def _container_runtime():
    """Return 'docker' if available, 'podman' if not, or raise if neither is found."""
    for runtime in ('docker', 'podman'):
        if shutil.which(runtime):
            return runtime
    print('Error: neither docker nor podman found on PATH.', file=sys.stderr)
    sys.exit(1)


def result_exists(base_config, benchmark, results_dir):
    """Return True if a completed result directory exists for this (base_config, benchmark) pair."""
    benchmark_text = benchmark if len(benchmark) <= 100 else benchmark[:100] + '__etc'
    suffix = f"{'+'.join(base_config)}_{benchmark_text}"
    if not os.path.isdir(results_dir):
        return False
    for entry in os.listdir(results_dir):
        if entry.startswith('results_') and entry.endswith(suffix):
            if os.path.isfile(os.path.join(results_dir, entry, 'executioninfo.txt')):
                return True
    return False


def build_image():
    print(f'Running make -C ./docker build-experiments')
    subprocess.run(
        ['make', '-C', f"{HERE}/docker", 'build-experiments'],
        check=True,
    )
    print('Build complete.')


def _docker_cmd(base_config, benchmark, results_dir):
    # Override the image entrypoint and call run() directly with the specific
    # base_config and benchmark, so each container runs exactly one simulation.
    snippet = (
        "import sys, os; "
        "os.chdir('/hotsniper'); "
        "sys.path.insert(0, '/hotsniper/simulationcontrol'); "
        f"from run import run; "
        f"run({base_config!r}, {benchmark!r}, ignore_error=True)"
    )
    return [
        _container_runtime(), 'run',
        '--privileged',
        '--rm',
        '-v', f'{results_dir}:/hotsniper/results',
        '--entrypoint', 'python3',
        DOCKER_EXPERIMENT_IMAGE,
        '-c', snippet,
    ]


def launch(experiments, results_dir, session, dry_run=False, skip_duplicates=False):
    if skip_duplicates:
        pending, skipped = [], []
        for exp in experiments:
            (skipped if result_exists(exp[1], exp[2], results_dir) else pending).append(exp)
        if skipped:
            print(f'Skipping {len(skipped)} already-completed experiment(s):')
            for label, _, _ in skipped:
                print(f'  [{label}]')
        experiments = pending

    n = len(experiments)
    if n == 0:
        print('No experiments to run — all already completed.')
        return

    if dry_run:
        print(f'Would create tmux session "{session}" with {n} windows:\n')
        for label, base_config, benchmark in experiments:
            cmd = ' '.join(shlex.quote(a) for a in _docker_cmd(base_config, benchmark, results_dir))
            print(f'  [{label}]\n  {cmd}\n')
        return

    if not shutil.which('tmux'):
        print('Error: tmux is not installed or not on PATH.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)

    # Kill any existing session with the same name
    subprocess.run(['tmux', 'kill-session', '-t', session], capture_output=True)

    # Create detached session with the first window named after the first experiment
    first_label = experiments[0][0]
    subprocess.run(
        ['tmux', 'new-session', '-d', '-s', session, '-n', first_label, '-x', '250', '-y', '50'],
        check=True,
    )

    # Create one window per remaining experiment
    for label, _, _ in experiments[1:]:
        subprocess.run(['tmux', 'new-window', '-t', session, '-n', label], check=True)

    # Fire each docker command into its window
    for i, (label, base_config, benchmark) in enumerate(experiments):
        cmd = ' '.join(shlex.quote(a) for a in _docker_cmd(base_config, benchmark, results_dir))
        subprocess.run(['tmux', 'send-keys', '-t', f'{session}:{i}', cmd, 'Enter'])

    # Start on window 0
    subprocess.run(['tmux', 'select-window', '-t', f'{session}:0'])

    print(f'Launched {n} experiments in tmux session "{session}" ({n} windows).')
    print('Attaching... (detach with Ctrl-b d, switch windows with Ctrl-b n/p or Ctrl-b <number>)')
    subprocess.run(['tmux', 'attach-session', '-t', session])


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments in parallel Docker containers with a live tmux view.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--build', action='store_true',
                        help='Rebuild the experiment Docker image before launching')
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR, metavar='DIR',
                        help=f'Host directory mounted as /hotsniper/results (default: {DEFAULT_RESULTS_DIR})')
    parser.add_argument('--session', default=DEFAULT_SESSION, metavar='NAME',
                        help=f'tmux session name (default: {DEFAULT_SESSION})')
    parser.add_argument('--list', action='store_true',
                        help='List experiments and exit without launching')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the docker commands that would run, without executing')
    parser.add_argument('--skip-duplicates', action='store_true',
                        help='Skip experiments that already have a completed result in --results-dir')
    parser.add_argument('--training', action='store_true',
                        help='Run training-data collection (maxFreq at 2.0–4.0 GHz) instead of policy evaluation')
    args = parser.parse_args()

    experiments = get_training_experiments() if args.training else get_experiments()

    if args.list:
        print(f'{len(experiments)} experiments:')
        for i, (label, base_config, benchmark) in enumerate(experiments, 1):
            print(f'  {i:2d}. [{label}]')
            print(f'       config:    {base_config}')
            print(f'       benchmark: {benchmark}')
        return

    if args.build:
        build_image()

    launch(experiments, args.results_dir, args.session,
           dry_run=args.dry_run, skip_duplicates=args.skip_duplicates)


if __name__ == '__main__':
    main()
