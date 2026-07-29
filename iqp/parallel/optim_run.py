import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from iqp.experiments import optim


def ckpt_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "results", "optim")
    os.makedirs(d, exist_ok=True)
    return d


def ckpt_path(opt, n):
    return os.path.join(ckpt_dir(), f"{opt}__n{n}.json")


def done(opt, n, nprob):
    p = ckpt_path(opt, n)
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        return isinstance(d, list) and len(d) == nprob and all(x is not None for x in d)
    except Exception:
        return False


def save(opt, n, values):
    p = ckpt_path(opt, n)
    json.dump(values, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", default=None, help="comma list, default all")
    ap.add_argument("--qubits", default=None, help="comma list, default all")
    ap.add_argument("--nprob", type=int, default=optim.N_PROBLEM)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = ap.parse_args()

    opts = args.opts.split(",") if args.opts else optim.OPTIMIZERS
    qubits = [int(x) for x in args.qubits.split(",")] if args.qubits else optim.QUBITS

    tasks = [(o, n, j)
             for o in opts for n in qubits
             if args.force or not done(o, n, args.nprob)
             for j in range(args.nprob)]
    print(f"optim scan: opts={opts} qubits={qubits} nprob={args.nprob} "
          f"-> {len(tasks)} tasks on {args.workers} workers", flush=True)

    buf = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(optim.compute_one, t) for t in tasks]
        for fut in as_completed(futs):
            (o, n, j), r = fut.result()
            d = buf[(o, n)]; d[j] = r
            if len(d) == args.nprob:
                save(o, n, [d[i] for i in range(args.nprob)])
                del buf[(o, n)]
                print(f"  done {o}/n{n}", flush=True)


if __name__ == "__main__":
    main()
