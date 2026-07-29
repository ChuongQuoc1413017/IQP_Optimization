import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from iqp.experiments import depth


def ckpt_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "results", "depth")
    os.makedirs(d, exist_ok=True)
    return d


def ckpt_path(metric, L):
    return os.path.join(ckpt_dir(), f"{metric}__L{L}.json")


def done(metric, L, nprob):
    p = ckpt_path(metric, L)
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return isinstance(d, list) and len(d) == nprob and all(x is not None for x in d)
    except Exception:
        return False


def save(metric, L, values):
    p = ckpt_path(metric, L)
    json.dump(values, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["ratio", "gv", "both"], default="both")
    ap.add_argument("--layers", default=None, help="comma list, e.g. 1,4,10")
    ap.add_argument("--nprob", type=int, default=depth.N_PROBLEM)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = ap.parse_args()

    metrics = ["ratio", "gv"] if args.metric == "both" else [args.metric]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else depth.LAYERS

    tasks = [(m, L, j)
             for m in metrics for L in layers
             if args.force or not done(m, L, args.nprob)
             for j in range(args.nprob)]
    print(f"depth scan: metrics={metrics} layers={layers} nprob={args.nprob} "
          f"-> {len(tasks)} tasks on {args.workers} workers", flush=True)

    buf = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(depth.compute_one, t) for t in tasks]
        for fut in as_completed(futs):
            (m, L, j), v = fut.result()
            d = buf[(m, L)]; d[j] = v
            if len(d) == args.nprob:
                save(m, L, [d[i] for i in range(args.nprob)])
                del buf[(m, L)]
                print(f"  done {m}/L{L}", flush=True)


if __name__ == "__main__":
    main()
