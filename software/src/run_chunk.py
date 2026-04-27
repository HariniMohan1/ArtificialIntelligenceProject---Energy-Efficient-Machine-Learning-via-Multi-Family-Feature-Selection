"""Run a single (dataset_idx, model_idx) chunk and append to results.csv."""
import sys, os, json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
os.environ["FAST_MODE"] = "1"

from run_experiments import (
    load_datasets,
    model_factory,
    run_one,
    SELECTORS,
    RESULTS_DIR,
)

ds_idx = int(sys.argv[1])
model_idx = int(sys.argv[2])
k_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

datasets = load_datasets()
models = model_factory()
mlist = list(models.items())

dataset = datasets[ds_idx]
model_name, model_ctor = mlist[model_idx]

rows = []
for sname, (sfam, sctor) in SELECTORS.items():
    if sname == "Baseline (no FS)":
        r = run_one(dataset, model_name, model_ctor, sname, sfam, None, 1.0)
    else:
        r = run_one(dataset, model_name, model_ctor, sname, sfam, sctor, k_ratio)
    rows.append(r)
    print(
        f"{dataset.name[:18]:18s} | {model_name[:20]:20s} | "
        f"{sname[:24]:24s} | acc={r['accuracy_mean']:.4f} | "
        f"t={r['total_time_s_mean']*1000:.0f}ms"
    )

out = RESULTS_DIR / f"chunk_d{ds_idx}_m{model_idx}_k{int(k_ratio*100)}.json"
with open(out, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Saved: {out}")
