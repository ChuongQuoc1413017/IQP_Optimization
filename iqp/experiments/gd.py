from iqp.core.problems import ising_obs, maxcut_obs, number_partition_obs, binary_ops_to_hamiltonian
from iqp.core.data import dataset
from iqp.core.engine import grad_expval, _seed_for
from iqp.core.ansatz import n_params_for, init_params

import numpy as np

N_TRIALS = 100

def gradient_variance(H, ops, coeffs, n, ansatz, spec, init, ham, j, n_trials=N_TRIALS):
    np.random.seed(_seed_for(init, ham, ansatz, n, j))
    npar = n_params_for(ansatz, spec, n)
    grads = []
    for _ in range(n_trials):
        p = init_params(init, npar, n)
        grads.append(grad_expval(p, H, ops, coeffs, n, ansatz, spec))
    grads = np.array(grads)
    return float(np.mean(np.var(grads, axis=0)))

def gd_ising(n, j, init, ansatz, spec):
    coeffs = np.array(dataset['ising'][n][j])
    ops = ising_obs(n, 2)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    return gradient_variance(H, ops, coeffs, n, ansatz, spec, init, 'ising', j)

def gd_maxcut(n, j, init, ansatz, spec):
    edges, weights = dataset['maxcut'][n][j]
    ops, coeffs = maxcut_obs(n, edges, weights)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    return gradient_variance(H, ops, coeffs, n, ansatz, spec, init, 'maxcut', j)

def gd_partition(n, j, init, ansatz, spec):
    numbers = dataset['partition'][n][j]
    ops, coeffs = number_partition_obs(numbers)
    H = binary_ops_to_hamiltonian(ops, coeffs)
    return gradient_variance(H, ops, coeffs, n, ansatz, spec, init, 'partition', j)

gd_fns = {"ising": gd_ising, "maxcut": gd_maxcut, "partition": gd_partition}

def compute_one(task):
    init, ham, ansatz, spec, n, j = task
    v = gd_fns[ham](n, j, init, ansatz, spec)
    return task, v
