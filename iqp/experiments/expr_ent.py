import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pennylane as qp
from scipy.spatial.distance import jensenshannon

from iqp.core.ansatz import apply_ansatz, n_params_for
from iqp.core.data import dataset
from iqp.core.problems import ising_obs, maxcut_obs, number_partition_obs

N_STATES = 1000
BINS = 150
QUBITS = [3, 6, 9, 12, 15]
HEA_LAYERS = 2
QAOA_P = 2
INSTANCE = 0
SV_TOL = 1e-15

CONFIGS = [
    ("single",         "iqp",  "single",     None),
    ("circular",       "iqp",  "circular",   None),
    ("full",           "iqp",  "full",       None),
    ("hea",            "hea",  HEA_LAYERS,   None),
    ("qaoa-ising",     "qaoa", QAOA_P,       "ising"),
    ("qaoa-maxcut",    "qaoa", QAOA_P,       "maxcut"),
    ("qaoa-partition", "qaoa", QAOA_P,       "partition"),
]

LABELS = [c[0] for c in CONFIGS]

def problem_ops(ham, n, j=INSTANCE):
    """Hamiltonian terms for the QAOA cost layer, from problem instance j."""
    if ham == "ising":
        return ising_obs(n, 2), np.array(dataset["ising"][n][j])
    if ham == "maxcut":
        edges, weights = dataset["maxcut"][n][j]
        return maxcut_obs(n, edges, weights)
    if ham == "partition":
        return number_partition_obs(dataset["partition"][n][j])
    raise ValueError(f"Invalid Hamiltonian: {ham!r}")

def _seed_for(label, n):
    key = f"expr_ent|{label}|{n}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

def build_state_circuit(n, ansatz, spec, ops, coeffs):
    dev = qp.device("lightning.qubit", wires=n, shots=None)

    @qp.qnode(dev)
    def state_circuit(params):
        apply_ansatz(params, ops, coeffs, n, ansatz, spec)
        return qp.state()

    return state_circuit

def random_states(n, ansatz, spec, ops, coeffs, n_states):
    """n_states states with angles drawn Uniform(0, pi), as in the notebook."""
    circuit = build_state_circuit(n, ansatz, spec, ops, coeffs)
    n_params = n_params_for(ansatz, spec, n)
    out = np.empty((n_states, 2 ** n), dtype=np.complex128)
    for i in range(n_states):
        out[i] = np.asarray(circuit(np.random.uniform(0, np.pi, n_params)))
    return out

def fidelity_distribution(states):
    """All pairwise |<psi_i|psi_j>|^2 for i < j, via one Gram matrix."""
    gram = states.conj() @ states.T
    iu = np.triu_indices(len(states), k=1)
    return np.abs(gram[iu]) ** 2

def expressibility(states, n, bins=BINS):
    fidelities = fidelity_distribution(states)

    hist, edges = np.histogram(fidelities, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    d = 2 ** n
    haar_pdf = (d - 1) * (1 - centers) ** (d - 2)

    hist = hist / np.sum(hist)
    haar_pdf = haar_pdf / np.sum(haar_pdf)

    return float(jensenshannon(hist + 1e-12, haar_pdf + 1e-12))

def entanglement_entropy(state, n):
    """Von Neumann entropy (base 2) of wires 0..n//2-1, via Schmidt values.

    Wire 0 is the most significant index in PennyLane's state vector, so
    reshaping to (2^cut, 2^(n-cut)) puts subsystem A on the row index.
    """
    cut = n // 2
    m = np.asarray(state).reshape(2 ** cut, 2 ** (n - cut))
    sv = np.linalg.svd(m, compute_uv=False)
    p = sv ** 2
    p = p[p > SV_TOL]
    return float(-np.sum(p * np.log2(p)))

def average_entropy(states, n):
    entropies = [entanglement_entropy(s, n) for s in states]
    return float(np.mean(entropies)), float(np.std(entropies))

def compute_one(task):
    """task = (label, ansatz, spec, ham, n, n_states, bins) -> (task, result)."""
    label, ansatz, spec, ham, n, n_states, bins = task

    ops, coeffs = problem_ops(ham, n) if ham is not None else (None, None)

    np.random.seed(_seed_for(label, n))
    t0 = time.time()
    states = random_states(n, ansatz, spec, ops, coeffs, n_states)

    expr = expressibility(states, n, bins)
    ent_mean, ent_std = average_entropy(states, n)

    return task, {
        "expr": expr,
        "entropy_mean": ent_mean,
        "entropy_std": ent_std,
        "n_states": n_states,
        "bins": bins,
        "seconds": round(time.time() - t0, 2),
    }

def ckpt_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "results", "expr_ent")
    os.makedirs(d, exist_ok=True)
    return d

def ckpt_path(label, n):
    return os.path.join(ckpt_dir(), f"{label}__n{n}.json")

def config_done(label, n, n_states, bins):
    p = ckpt_path(label, n)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return d.get("n_states") == n_states and d.get("bins") == bins
    except Exception:
        return False

def save_ckpt(label, n, result):
    p = ckpt_path(label, n)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, p)

def merge(qubits, labels=LABELS):
    """Collect checkpoints into the notebook's {metric: {label: {n: v}}} shape."""
    merged = {"expr": {}, "entropy_mean": {}, "entropy_std": {}}
    missing = []
    for label in labels:
        for k in merged:
            merged[k][label] = {}
        for n in qubits:
            p = ckpt_path(label, n)
            if not os.path.exists(p):
                missing.append((label, n))
                continue
            with open(p) as f:
                d = json.load(f)
            for k in merged:
                merged[k][label][str(n)] = d[k]
    return merged, missing

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8,
                    help="processes; each holds n_states x 2^n complex128")
    ap.add_argument("--n-states", type=int, default=N_STATES)
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--qubits", type=int, nargs="+", default=QUBITS)
    ap.add_argument("--labels", nargs="+", default=LABELS, choices=LABELS)
    ap.add_argument("--force", action="store_true", help="ignore checkpoints")
    args = ap.parse_args()

    configs = [c for c in CONFIGS if c[0] in args.labels]
    tasks = [(label, ansatz, spec, ham, n, args.n_states, args.bins)
             for (label, ansatz, spec, ham) in configs
             for n in args.qubits
             if args.force or not config_done(label, n, args.n_states, args.bins)]

    # Largest first: cost scales as 2^n, so start the stragglers early.
    tasks.sort(key=lambda t: -t[4])

    total = len(configs) * len(args.qubits)
    peak = args.workers * args.n_states * 2 ** max(args.qubits) * 16 / 2 ** 30
    print(f"{total} configs, {len(tasks)} to run, {args.workers} workers "
          f"(worst-case resident ~{peak:.1f} GB)", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(compute_one, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            task, res = fut.result()
            label, n = task[0], task[4]
            save_ckpt(label, n, res)
            print(f"  [{k}/{len(tasks)}] {label}/n{n}  "
                  f"expr={res['expr']:.4f}  S={res['entropy_mean']:.4f}  "
                  f"({res['seconds']}s)", flush=True)

    merged, missing = merge(args.qubits, args.labels)
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "expr_ent_merged.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=4)

    if missing:
        print(f"missing {len(missing)}: {missing[:8]}")
    print(f"wrote {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
