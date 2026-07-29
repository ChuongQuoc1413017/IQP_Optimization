import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse, json, time, importlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from iqp.parallel import grid

def ckpt_dir(experiment):
    d = os.path.join(os.path.dirname(__file__), "..", "results", experiment)
    os.makedirs(d, exist_ok=True)
    return d

def ckpt_path(experiment, init, ham, label, n):
    return os.path.join(ckpt_dir(experiment), f"{init}__{ham}__{label}__n{n}.json")

def config_done(experiment, init, ham, label, n):
    p = ckpt_path(experiment, init, ham, label, n)
    if not os.path.exists(p): return False
    try:
        with open(p) as f: data = json.load(f)
        return isinstance(data, list) and len(data) == grid.N_PROBLEM and all(x is not None for x in data)
    except Exception:
        return False

def save_ckpt(experiment, init, ham, label, n, ratios):
    p = ckpt_path(experiment, init, ham, label, n)
    tmp = p + ".tmp"
    with open(tmp, "w") as f: json.dump(ratios, f)
    os.replace(tmp, p)         

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["ratio", "gd"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = ap.parse_args()

    exp = importlib.import_module(f"iqp.experiments.{args.experiment}")   
    compute_one = exp.compute_one

    configs = grid.shard_configs(args.shard, args.nshards)              
    todo = [c for c in configs
            if args.force or not config_done(args.experiment, c[0], c[1], grid.config_label(c[2], c[3]), c[4])]

    tasks = [(init, ham, ansatz, spec, n, j)
             for (init, ham, ansatz, spec, n) in todo
             for j in range(grid.N_PROBLEM)]
    print(f"shard {args.shard}/{args.nshards}: {len(configs)} configs, "
          f"{len(todo)} inprogress, {len(tasks)} tasks, {args.workers} workers")
    
    buf = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(compute_one, t) for t in tasks]
        for fut in as_completed(futs):
            (init, ham, ansatz, spec, n, j), r = fut.result()
            key = (init, ham, ansatz, spec, n)
            d = buf[key]; d[j] = r
            if len(d) == grid.N_PROBLEM:                    
                ratios = [d[i] for i in range(grid.N_PROBLEM)]
                save_ckpt(args.experiment, init, ham, grid.config_label(ansatz, spec), n, ratios)
                del buf[key]
                print(f"  done {init}/{ham}/{grid.config_label(ansatz,spec)}/n{n}", flush=True)

if __name__ == "__main__":
    main()