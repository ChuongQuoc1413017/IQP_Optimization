import numpy as np
import pennylane as qp
from pennylane import numpy as qnp
from scipy.optimize import minimize

from iqp.core.problems import ising_obs, binary_ops_to_hamiltonian
from iqp.core.data import dataset, exact_ising
from iqp.core.ansatz import apply_ansatz, n_params_for, init_params
from iqp.core.engine import _seed_for, N_ITERS

INIT, HAM, MODE = "normal", "ising", "full"
QUBITS = [3, 6, 9, 12, 15]
OPTIMIZERS = ["cobyla", "spsa", "sgd", "qng"]   
N_PROBLEM = 50

SGD_LR = 1e-3   
QNG_LR = 1e-3

def _qnode(H, n):
    dev = qp.device("lightning.qubit", wires=n, shots=None)

    def _expval(params):
        apply_ansatz(params, None, None, n, "iqp", MODE)
        return qp.expval(H)

    return qp.QNode(_expval, dev, interface="autograd")

def _init(n, j):
    np.random.seed(_seed_for(INIT, HAM, "iqp", n, j))
    return init_params(INIT, n_params_for("iqp", MODE, n), n)

def _final_loss(opt_name, qn, p0):
    if opt_name == "cobyla":
        fun = lambda x: float(qn(qnp.array(x, requires_grad=False)))
        res = minimize(fun, np.array(p0), method="COBYLA",
                       options={"maxiter": N_ITERS})
        return float(res.fun)

    p = qnp.array(p0, requires_grad=True)
    if opt_name == "sgd":
        opt = qp.GradientDescentOptimizer(SGD_LR)
    elif opt_name == "qng":
        opt = qp.QNGOptimizer(QNG_LR)
    elif opt_name == "spsa":
        opt = qp.SPSAOptimizer(maxiter=N_ITERS)
    else:
        raise ValueError(f"Unknown optimizer {opt_name!r}")
    for _ in range(N_ITERS):
        p = opt.step(qn, p)
    return float(qn(p))

def ratio_one(opt_name, n, j):
    coeffs = np.array(dataset[HAM][n][j])
    H = binary_ops_to_hamiltonian(ising_obs(n, 2), coeffs)
    qn = _qnode(H, n)
    final_loss = _final_loss(opt_name, qn, _init(n, j))
    return float((exact_ising[n][j] - final_loss) / exact_ising[n][j])


def compute_one(task):
    """task = (opt_name, n, j) -> (task, r_RA)."""
    opt_name, n, j = task
    return task, ratio_one(opt_name, n, j)
