import numpy as np
import pennylane as qp
from pennylane import numpy as qnp

from iqp.core.problems import ising_obs, binary_ops_to_hamiltonian
from iqp.core.data import dataset, exact_ising
from iqp.core.ansatz import apply_iqp_layers, n_params_for, init_params
from iqp.core.engine import _seed_for, N_ITERS, STEPSIZE

INIT, HAM, MODE, N = "normal", "ising", "full", 12
LAYERS = [1, 2, 3, 4, 6, 8, 10]
N_PROBLEM = 50
GV_TRIALS = 100         


def _hamiltonian(j):
    coeffs = np.array(dataset[HAM][N][j])
    ops = ising_obs(N, 2)
    return binary_ops_to_hamiltonian(ops, coeffs)


def _qnode(H, n_layers):
    dev = qp.device("lightning.qubit", wires=N, shots=None)

    def _expval(params):
        apply_iqp_layers(params, N, MODE, n_layers)
        return qp.expval(H)

    return qp.QNode(_expval, dev, interface="autograd")


def _init(n_layers, j):
    np.random.seed(_seed_for(INIT, HAM, "iqp", N, j))
    return init_params(INIT, n_layers * n_params_for("iqp", MODE, N), N)


def ratio_one(n_layers, j):
    H = _hamiltonian(j)
    qnode = _qnode(H, n_layers)
    params = qnp.array(_init(n_layers, j), requires_grad=True)
    opt = qp.AdamOptimizer(STEPSIZE)
    for _ in range(N_ITERS):
        params = opt.step(qnode, params)
    final_loss = float(qnode(params))
    return float((exact_ising[N][j] - final_loss) / exact_ising[N][j])


def gradvar_one(n_layers, j):
    H = _hamiltonian(j)
    qnode = _qnode(H, n_layers)
    grad_fn = qp.grad(qnode)
    np.random.seed(_seed_for(INIT, HAM, "iqp", N, j))
    npar = n_layers * n_params_for("iqp", MODE, N)
    grads = [np.array(grad_fn(qnp.array(init_params(INIT, npar, N), requires_grad=True)))
             for _ in range(GV_TRIALS)]
    return float(np.mean(np.var(np.array(grads), axis=0)))


def compute_one(task):
    """task = (metric, L, j) -> (task, value)."""
    metric, L, j = task
    v = ratio_one(L, j) if metric == "ratio" else gradvar_one(L, j)
    return task, v