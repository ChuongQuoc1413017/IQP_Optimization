import numpy as np
from iqp.core.problems import ising_obs, maxcut_obs, number_partition_obs, binary_ops_to_hamiltonian, maxcut_value, solve_maxcut_exact_symmetry, sum_selected
from iqp.core.data import dataset, exact_ising
from iqp.core.engine import train_exact, get_probs, _seeded_init
from iqp.core.ansatz import n_params_for
from numberpartitioning import karmarkar_karp

def ratio_ising(n, j, init, ansatz, spec):
    coeffs = np.array(dataset['ising'][n][j])
    ops = ising_obs(n, 2)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "ising", ansatz, j)
    _, final_loss = train_exact(H, ops, coeffs, n, ansatz, spec, p0)
    return float((exact_ising[n][j] - final_loss) / exact_ising[n][j])

def ratio_maxcut(n, j, init, ansatz, spec):
    edges, weights = dataset['maxcut'][n][j]
    ops, coeffs = maxcut_obs(n, edges, weights)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "maxcut", ansatz, j)
    params, _ = train_exact(H, ops, coeffs, n, ansatz, spec, p0)
    probs = get_probs(params, ops, coeffs, n, ansatz, spec)
    bits = np.binary_repr(int(np.argmax(probs)), n)
    sol = maxcut_value(bits, edges, weights)
    ex = solve_maxcut_exact_symmetry(edges, weights)[0]
    return float(abs(sol - ex) / ex)

def ratio_partition(n, j, init, ansatz, spec):
    numbers = dataset['partition'][n][j]
    ops, coeffs = number_partition_obs(numbers)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    p0 = _seeded_init(init, n_params_for(ansatz, spec, n), n, "partition", ansatz, j)
    params, _ = train_exact(H, ops, coeffs, n, ansatz, spec, p0)
    probs = get_probs(params, ops, coeffs, n, ansatz, spec)
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
