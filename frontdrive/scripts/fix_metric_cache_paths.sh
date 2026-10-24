#!/usr/bin/env bash
# Rewrite the absolute paths stored in a metric cache's metadata CSV(s) to the cache's current location.
# Works for any cache dir name (metric_cache/, train_metric_cache/, metric_cache_navtrain_val_official/).
# MetricCacheLoader reads only the FIRST metadata csv, so several csvs are merged into one first.
# Usage: bash scripts/fix_metric_cache_paths.sh <metric_cache_dir>      (idempotent)
set -euo pipefail
MC=$(readlink -f "${1:?metric cache dir}"); name=$(basename "$MC"); META=$MC/metadata
ls "$META"/*.csv >/dev/null 2>&1 || { echo "no metadata csv under $META"; exit 1; }
if [ "$(ls "$META"/*.csv | wc -l)" -gt 1 ]; then
  mkdir -p "$META/_parts"
  python3 - "$META" <<'PY'
import glob, os, shutil, sys
meta = sys.argv[1]; files = sorted(glob.glob(os.path.join(meta, "*.csv")))
header = None; bodies = []
for f in files:
    with open(f) as fh:
        h = fh.readline(); bodies.append(fh.read())
    header = header or h
with open(os.path.join(meta, "merged_metadata.csv"), "w") as out:
    out.write(header); out.write("".join(bodies))
for f in files:
    shutil.move(f, os.path.join(meta, "_parts", os.path.basename(f)))
print("merged", len(files), "metadata csv files into merged_metadata.csv")
PY
fi
csv=$(ls "$META"/*.csv | head -1)
old=$(sed -n 2p "$csv" | grep -oE "[^,\"]*/$name/" | head -1 | sed -E "s|/$name/\$|/$name|")
[ -n "$old" ] || { echo "no '/$name/' path found in line 2 of $csv"; exit 1; }
echo "old root: $old"; echo "new root: $MC"
if [ "$old" != "$MC" ]; then for f in "$META"/*.csv; do sed -i "s|$old|$MC|g" "$f"; done; echo "rewritten $(ls "$META"/*.csv | wc -l) csv"; else echo "paths already current"; fi
python3 - "$csv" "$MC" <<'PY'
import csv, os, sys
path_csv, root = sys.argv[1], sys.argv[2]
n = miss = 0; example = None
with open(path_csv) as fh:
    r = csv.reader(fh); next(r)
    for row in r:
        for c in row:
            if c.startswith(root + "/"):
                n += 1
                if not os.path.exists(c):
                    miss += 1; example = example or c
                break
print(f"rows with paths: {n}; missing files: {miss}" + (f"; e.g. {example}" if example else ""))
sys.exit(2 if (n == 0 or miss) else 0)
PY
