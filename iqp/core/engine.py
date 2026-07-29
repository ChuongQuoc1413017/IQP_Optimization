import numpy as np
import pennylane as qp
from pennylane import numpy as qnp

from iqp.core.ansatz import apply_ansatz, init_params

import hashlib

N_ITERS = 2500   
STEPSIZE = 1e-3

def _expval(params, H, ops, coeffs, n_qubits, ansatz, spec):
    apply_ansatz(params, ops, coeffs, n_qubits, ansatz, spec)
    return qp.expval(H)

def _probs(params, ops, coeffs, n_qubits, ansatz, spec):
    apply_ansatz(params, ops, coeffs, n_qubits, ansatz, spec)
    return qp.probs(wires=range(n_qubits))

def train_exact(H, ops, coeffs, n_qubits, ansatz, spec, params_init,
                  n_iters=N_ITERS, stepsize=STEPSIZE):
    dev = qp.device("lightning.qubit", wires=n_qubits, shots=None)
    qnode = qp.QNode(_expval, dev, interface="autograd")
    params = qnp.array(params_init, requires_grad=True)
    opt = qp.AdamOptimizer(stepsize)
    cost = lambda p: qnode(p, H, ops, coeffs, n_qubits, ansatz, spec)
    for _ in range(n_iters):
        params = opt.step(cost, params)
    return params, float(cost(params))

def get_probs(params, ops, coeffs, n_qubits, ansatz, spec):
    dev = qp.device("lightning.qubit", wires=n_qubits, shots=None)
    qnode = qp.QNode(_probs, dev, interface="autograd")
    return np.array(qnode(qnp.array(params, requires_grad=False),
                          ops, coeffs, n_qubits, ansatz, spec))

def _seed_for(init, ham, ansatz, n, j):
    key = f"{init}|{ham}|{ansatz}|{n}|{j}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")

def _seeded_init(init, n_params, n, ham, ansatz, j):
    np.random.seed(_seed_for(init, ham, ansatz, n, j))
    return init_params(init, n_params, n)

def grad_expval(params, H, ops, coeffs, n_qubits, ansatz, spec):
    dev = qp.device("lightning.qubit", wires=n_qubits, shots=None)
    qnode = qp.QNode(_expval, dev, interface="autograd")
    cost = lambda p: qnode(p, H, ops, coeffs, n_qubits, ansatz, spec)
    grad_fn = qp.grad(cost)
    g = grad_fn(qnp.array(params, requires_grad=True))
    return np.array(g)
