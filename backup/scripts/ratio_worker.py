"""Worker cho chay song song HEA/QAOA ratio (bam dung ratio_ising/maxcut/partition
cua Ratio_5_Initializations_Extra.ipynb).

- Ep 1 luong/worker (tranh oversubscription khi chay ProcessPool).
- Seed TAT DINH theo tung task (hashlib -> tai lap duoc xuyen tien trinh, doc lap
  giua cac worker). KHAC voi ban goc dung np.random.seed(42) mot lan.
- Ham compute_one(task) la thu tuc goi tu ProcessPoolExecutor. Task:
      (init, ham, ansatz, spec, n, j)
  tra ve (task, ratio_float).
"""

import os
# PHAI dat truoc khi import numpy: khop toc do voi kernel don luong dang chay
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
import hashlib

import numpy as np
import pennylane as qp
from pennylane import numpy as qnp

# --- Cho phep import cac module cung thu muc du chay o cwd khac ---
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ansatz_extra import apply_ansatz, n_params_for, init_params
from IQP_circuit_pennylane import binary_ops_to_hamiltonian
from IQP_circuit import maxcut_obs, number_partition_obs
from MaxCut import maxcut_value, solve_maxcut_exact_symmetry
from Partition import sum_selected
from numberpartitioning import karmarkar_karp

# ---- Budget optimizer (giong file IQP/Extra) ----
N_ITERS = 1000
STEPSIZE = 1e-3

# ---- Load datasets 1 lan moi worker (o module level) ----
with open(os.path.join(_HERE, "ising.json"), "r") as f:
    dataset_ising = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(_HERE, "ising_loss_exact.json"), "r") as f:
    exact_ising = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(_HERE, "maxcut.json"), "r") as f:
    dataset_maxcut = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(_HERE, "partition.json"), "r") as f:
    dataset_partition = {int(k): v for k, v in json.load(f).items()}


def ising_obs_local(n_qubits, corr=2):
    obs = []
    for i in range(n_qubits):
        v = [0] * n_qubits; v[i] = 1; obs.append(v)
    if corr == 2:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                v = [0] * n_qubits; v[i] = 1; v[j] = 1; obs.append(v)
    return np.array(obs)


# ---- Optimization core (giong train_exact_a / get_probs_a ban Extra) ----
def _expval_flat_a(params, H, ops, coeffs, n_qubits, ansatz, spec):
    apply_ansatz(params, ops, coeffs, n_qubits, ansatz, spec)
    return qp.expval(H)


def _probs_flat_a(params, ops, coeffs, n_qubits, ansatz, spec):
    apply_ansatz(params, ops, coeffs, n_qubits, ansatz, spec)
    return qp.probs(wires=range(n_qubits))


def train_exact_a(H, ops, coeffs, n_qubits, ansatz, spec, params_init,
                  n_iters=N_ITERS, stepsize=STEPSIZE):
    dev = qp.device("lightning.qubit", wires=n_qubits, shots=None)
    qnode = qp.QNode(_expval_flat_a, dev, interface="autograd")
    params = qnp.array(params_init, requires_grad=True)
    opt = qp.AdamOptimizer(stepsize)
    cost = lambda p: qnode(p, H, ops, coeffs, n_qubits, ansatz, spec)
    for _ in range(n_iters):
        params = opt.step(cost, params)
    return params, float(cost(params))


def get_probs_a(params, ops, coeffs, n_qubits, ansatz, spec):
    dev = qp.device("lightning.qubit", wires=n_qubits, shots=None)
    qnode = qp.QNode(_probs_flat_a, dev, interface="autograd")
    return np.array(qnode(qnp.array(params, requires_grad=False),
                          ops, coeffs, n_qubits, ansatz, spec))


# ---- Seed TAT DINH theo task (thay cho np.random.seed(42) toan cuc) ----
def _seed_for(init, ham, ansatz, n, j):
    key = f"{init}|{ham}|{ansatz}|{n}|{j}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def _seeded_init(init, n_params, n, ham, ansatz, j):
    np.random.seed(_seed_for(init, ham, ansatz, n, j))
    return init_params(init, n_params, n)


# ---- Ratio functions (dinh nghia r_RA y het ban Extra; chi them seed tat dinh) ----
def ratio_ising(n, j, init, ansatz, spec):
    coeffs = np.array(dataset_ising[n][j])
    ops = ising_obs_local(n, 2)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "ising", ansatz, j)
    _, final_loss = train_exact_a(H, ops, coeffs, n, ansatz, spec, p0)
    return float((exact_ising[n][j] - final_loss) / exact_ising[n][j])


def ratio_maxcut(n, j, init, ansatz, spec):
    edges, weights = dataset_maxcut[n][j]
    ops, coeffs = maxcut_obs(n, edges, weights)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "maxcut", ansatz, j)
    params, _ = train_exact_a(H, ops, coeffs, n, ansatz, spec, p0)
    probs = get_probs_a(params, ops, coeffs, n, ansatz, spec)
    bits = np.binary_repr(int(np.argmax(probs)), n)
    sol = maxcut_value(bits, edges, weights)
    ex = solve_maxcut_exact_symmetry(edges, weights)[0]
    return float(abs(sol - ex) / ex)


def ratio_partition(n, j, init, ansatz, spec):
    numbers = dataset_partition[n][j]
    ops, coeffs = number_partition_obs(numbers)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "partition", ansatz, j)
    params, _ = train_exact_a(H, ops, coeffs, n, ansatz, spec, p0)
    probs = get_probs_a(params, ops, coeffs, n, ansatz, spec)
    bits = np.binary_repr(int(np.argmax(probs)), n)
    sol = sum_selected(bits, numbers)
    ex = karmarkar_karp(numbers, num_parts=2)
    denom = sum(ex.partition[0])
    return float(abs(sol - denom) / denom)


ratio_fns = {"ising": ratio_ising, "maxcut": ratio_maxcut, "partition": ratio_partition}


def compute_one(task):
    """task = (init, ham, ansatz, spec, n, j) -> (task, ratio_float)."""
    init, ham, ansatz, spec, n, j = task
    r = ratio_fns[ham](n, j, init, ansatz, spec)
    return task, r


# =====================================================================
# TRAJECTORY MODE (pilot "ratio vs iteration")
#
# Muc dich: phan xu undertraining vs local-minima -- ghi lai ratio tai
# cac moc iteration trong luc train thay vi chi gia tri cuoi.
# - Dinh nghia ratio Y HET ratio_ising/maxcut/partition o tren.
# - Seed init Y HET _seeded_init -> params khoi tao TRUNG voi ban
#   fixed-budget (so sanh truc tiep duoc: diem tai iter=1000 cua
#   trajectory ~ gia tri trong ratio_5inits_*.json).
# - KHONG dung cham API cu (compute_one).
# =====================================================================

def _train_ratio_traj(H, ops, coeffs, n_qubits, ansatz, spec, params_init,
                      ratio_at, n_iters, record_every, stepsize=STEPSIZE):
    """Train nhu train_exact_a nhung ghi ratio moi `record_every` buoc.

    ratio_at(params, cost, probs_fn) -> float: ratio tai params hien tai.
    Tra ve (iters, ratios); ghi tai moc 0, record_every, ... va moc cuoi.
    Chi phi them: 1-2 forward pass moi moc ghi (khong dang ke so voi
    n_iters buoc gradient).
    """
    dev = qp.device("lightning.qubit", wires=n_qubits, shots=None)
    expval_qnode = qp.QNode(_expval_flat_a, dev, interface="autograd")
    probs_qnode = qp.QNode(_probs_flat_a, dev, interface="autograd")

    def cost(p):
        return expval_qnode(p, H, ops, coeffs, n_qubits, ansatz, spec)

    def probs_fn(p):
        return np.array(probs_qnode(qnp.array(p, requires_grad=False),
                                    ops, coeffs, n_qubits, ansatz, spec))

    params = qnp.array(params_init, requires_grad=True)
    opt = qp.AdamOptimizer(stepsize)
    iters, ratios = [], []
    for it in range(n_iters + 1):
        if it % record_every == 0 or it == n_iters:
            iters.append(it)
            ratios.append(float(ratio_at(params, cost, probs_fn)))
        if it < n_iters:
            params = opt.step(cost, params)
    return iters, ratios


def traj_ising(n, j, init, ansatz, spec, n_iters, record_every):
    coeffs = np.array(dataset_ising[n][j])
    ops = ising_obs_local(n, 2)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    ex = exact_ising[n][j]
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "ising", ansatz, j)

    def ratio_at(params, cost, probs_fn):
        return (ex - float(cost(params))) / ex

    return _train_ratio_traj(H, ops, coeffs, n, ansatz, spec, p0,
                             ratio_at, n_iters, record_every)


def traj_maxcut(n, j, init, ansatz, spec, n_iters, record_every):
    edges, weights = dataset_maxcut[n][j]
    ops, coeffs = maxcut_obs(n, edges, weights)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    ex = solve_maxcut_exact_symmetry(edges, weights)[0]
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "maxcut", ansatz, j)

    def ratio_at(params, cost, probs_fn):
        bits = np.binary_repr(int(np.argmax(probs_fn(params))), n)
        return abs(maxcut_value(bits, edges, weights) - ex) / ex

    return _train_ratio_traj(H, ops, coeffs, n, ansatz, spec, p0,
                             ratio_at, n_iters, record_every)


def traj_partition(n, j, init, ansatz, spec, n_iters, record_every):
    numbers = dataset_partition[n][j]
    ops, coeffs = number_partition_obs(numbers)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    ex = karmarkar_karp(numbers, num_parts=2)
    denom = sum(ex.partition[0])
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "partition", ansatz, j)

    def ratio_at(params, cost, probs_fn):
        bits = np.binary_repr(int(np.argmax(probs_fn(params))), n)
        return abs(sum_selected(bits, numbers) - denom) / denom

    return _train_ratio_traj(H, ops, coeffs, n, ansatz, spec, p0,
                             ratio_at, n_iters, record_every)


traj_fns = {"ising": traj_ising, "maxcut": traj_maxcut, "partition": traj_partition}


def compute_one_traj(task):
    """task = (init, ham, ansatz, spec, n, j, n_iters, record_every)
    -> (task, iters, ratios)."""
    init, ham, ansatz, spec, n, j, n_iters, record_every = task
    iters, ratios = traj_fns[ham](n, j, init, ansatz, spec, n_iters, record_every)
    return task, iters, ratios
