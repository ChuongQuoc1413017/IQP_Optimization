"""PILOT DRIVER: ratio vs iteration (phan xu undertraining vs local-minima).

Chay `ratio_worker.compute_one_traj` song song, checkpoint theo tung config
(init, ham, ansatz, n) vao `ckpt_pilot/`, RESUME duoc neu ngat giua chung.

Luoi pilot (lat cat dai dien, KHONG phai full grid):
    ham    : ising, maxcut
    n      : 9, 15
    ansatz : iqp-single / iqp-circular / iqp-full / hea-2 / qaoa-2
    init   : normal, pi4, he, glorot
    inst   : N_INSTANCES instance dau (j = 0..N_INSTANCES-1)
    budget : N_ITERS iteration, ghi ratio moi RECORD_EVERY buoc

Seed init y het ratio_worker._seeded_init -> diem tai iter=1000 cua trajectory
so sanh truc tiep duoc voi gia tri fixed-budget trong ratio_5inits_*.json.

Cach dung:
    python pilot_traj_driver.py --smoke    # 1 task nho, kiem tra truoc khi mo pool
    python pilot_traj_driver.py            # chay pool (resume neu co checkpoint)
    python pilot_traj_driver.py --merge    # chi gop checkpoint -> pilot_traj.json

Ve hinh (chay duoc GIUA CHUNG, config nao xong ve config do):
    python pilot_traj_plot.py
"""

import os
import sys
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- Cau hinh pilot ----
N_ITERS      = 20000
RECORD_EVERY = 100
N_INSTANCES  = 10
MAX_WORKERS  = 8

INITS  = ["normal", "pi4", "he", "glorot"]
HAMS   = ["ising", "maxcut"]
ANSATZ = [("iqp", "single"), ("iqp", "circular"), ("iqp", "full"),
          ("hea", 2), ("qaoa", 2)]
QUBITS = [9, 15]

CKPT = os.path.join(_HERE, "ckpt_pilot")
OUT  = os.path.join(_HERE, "pilot_traj.json")


def akey(ansatz, spec):
    return f"{ansatz}-{spec}"


def ckpt_path(init, ham, ak, n):
    return os.path.join(CKPT, f"{init}__{ham}__{ak}__n{n}.json")


def config_done(init, ham, ak, n):
    p = ckpt_path(init, ham, ak, n)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return (d.get("n_iters") == N_ITERS
                and d.get("record_every") == RECORD_EVERY
                and len(d.get("ratios", [])) == N_INSTANCES)
    except Exception:
        return False


def save_ckpt(init, ham, ak, n, iters, ratios_by_j):
    """ratios_by_j: {j: [ratio tai tung moc]} du N_INSTANCES phan tu."""
    p = ckpt_path(init, ham, ak, n)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"n_iters": N_ITERS, "record_every": RECORD_EVERY,
                   "iters": iters,
                   "ratios": [ratios_by_j[j] for j in range(N_INSTANCES)]}, f)
    os.replace(tmp, p)  # atomic


def build_tasks():
    tasks, skipped = [], 0
    for init in INITS:
        for ham in HAMS:
            for ansatz, spec in ANSATZ:
                for n in QUBITS:
                    if config_done(init, ham, akey(ansatz, spec), n):
                        skipped += 1
                        continue
                    for j in range(N_INSTANCES):
                        tasks.append((init, ham, ansatz, spec, n, j,
                                      N_ITERS, RECORD_EVERY))
    return tasks, skipped


def merge():
    merged, missing = {}, []
    for init in INITS:
        merged[init] = {}
        for ham in HAMS:
            merged[init][ham] = {}
            for ansatz, spec in ANSATZ:
                ak = akey(ansatz, spec)
                merged[init][ham][ak] = {}
                for n in QUBITS:
                    p = ckpt_path(init, ham, ak, n)
                    if not os.path.exists(p):
                        missing.append((init, ham, ak, n))
                        continue
                    with open(p) as f:
                        merged[init][ham][ak][str(n)] = json.load(f)
    if missing:
        print(f"THIEU {len(missing)}/{len(INITS)*len(HAMS)*len(ANSATZ)*len(QUBITS)} "
              f"config -> CHUA xuat {os.path.basename(OUT)}. Vi du:", missing[:6])
        return False
    with open(OUT, "w") as f:
        json.dump(merged, f)
    print("Da xuat", OUT)
    return True


def smoke():
    from ratio_worker import compute_one_traj
    t0 = time.time()
    task = ("normal", "ising", "iqp", "full", 3, 0, 300, 50)
    _, iters, ratios = compute_one_traj(task)
    print(f"smoke ising  n=3 300it: {time.time()-t0:5.1f}s | "
          f"r: {ratios[0]:.3f} -> {ratios[-1]:.3f} | moc: {iters}")
    t0 = time.time()
    task = ("he", "maxcut", "hea", 2, 3, 0, 300, 50)
    _, iters, ratios = compute_one_traj(task)
    print(f"smoke maxcut n=3 300it: {time.time()-t0:5.1f}s | "
          f"r: {ratios[0]:.3f} -> {ratios[-1]:.3f}")
    # uoc luong toc do o n lon nhat de tinh ETA truoc khi mo pool
    t0 = time.time()
    task = ("normal", "ising", "iqp", "full", max(QUBITS), 0, 100, 100)
    compute_one_traj(task)
    per_it = (time.time() - t0) / 100
    n_runs = len(INITS) * len(HAMS) * len(ANSATZ) * len(QUBITS) * N_INSTANCES
    est = n_runs * N_ITERS * per_it / MAX_WORKERS / 3600
    print(f"toc do n={max(QUBITS)} full-IQP: {per_it*1000:.1f} ms/iter "
          f"-> uoc luong THO ca pilot ({n_runs} runs, {MAX_WORKERS} workers): "
          f"~{est:.1f} h (cac ansatz it tham so se nhanh hon)")


def main():
    os.makedirs(CKPT, exist_ok=True)
    if "--smoke" in sys.argv:
        smoke()
        return
    if "--merge" in sys.argv:
        merge()
        return

    from ratio_worker import compute_one_traj
    tasks, skipped = build_tasks()
    total_cfg = len(INITS) * len(HAMS) * len(ANSATZ) * len(QUBITS)
    remaining_cfg = len({(t[0], t[1], akey(t[2], t[3]), t[4]) for t in tasks})
    print(f"configs xong (bo qua): {skipped}/{total_cfg} | con lai: {remaining_cfg} "
          f"| tasks: {len(tasks)} | workers: {MAX_WORKERS}")
    if not tasks:
        merge()
        return

    buf = defaultdict(dict)   # (init,ham,ak,n) -> {j: ratios}
    iters_ref = {}            # (init,ham,ak,n) -> iters (giong nhau moi task)
    done_cfg, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(compute_one_traj, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            (init, ham, ansatz, spec, n, j, _, _), iters, ratios = fut.result()
            key = (init, ham, akey(ansatz, spec), n)
            buf[key][j] = ratios
            iters_ref[key] = iters
            if len(buf[key]) == N_INSTANCES:
                save_ckpt(*key, iters_ref[key], buf.pop(key))
                done_cfg += 1
                el = time.time() - t0
                eta = el / done_cfg * (remaining_cfg - done_cfg)
                finals = "da luu"
                print(f"[{el/60:6.1f}m] cfg {done_cfg}/{remaining_cfg} "
                      f"{key[0]:7s} {key[1]:7s} {key[2]:12s} n={key[3]:<2d} {finals} "
                      f"| task {k}/{len(tasks)} | ETA ~{eta/60:.0f}m", flush=True)
    print(f"\nXONG trong {(time.time()-t0)/3600:.2f} h")
    merge()


if __name__ == "__main__":
    main()
