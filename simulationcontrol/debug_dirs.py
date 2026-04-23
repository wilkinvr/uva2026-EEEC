import os, re
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
for d in sorted(os.listdir(RESULTS_DIR)):
    if '2026' not in d: continue
    # Format: results_YYYY-MM-DD_HH.MM_CONFIG_BENCHMARK
    m = re.match(r'results_(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2})_(.+)_(parsec-.+)', d)
    if m:
        print(f'config={m.group(2)!r}  bench={m.group(3)!r}')
    else:
        print(f'NO MATCH: {d!r}')
