import numpy as np
import pennylane as qp
from pennylane import numpy as qnp

# Tái dùng lớp IQP đã validate (không viết lại)
from IQP_circuit_pennylane import iqp_layer

def hea_layer(params, n_qubits: int, n_layers: int) -> None:
    k = 0  
    for _ in range(n_layers):
      
        for w in range(n_qubits):
            qp.RY(params[k], wires=w)
            k += 1
 
        for w in range(n_qubits):
            qp.CNOT(wires=[w, (w + 1) % n_qubits])


def n_params_hea(n_qubits: int, n_layers: int) -> int:
    return n_layers * n_qubits

def qaoa_layer(params, ops, coeffs, n_qubits: int, p: int) -> None:
    gammas = params[:p]
    betas = params[p:]

    for w in range(n_qubits):
        qp.Hadamard(wires=w)

    for k in range(p):
        g = gammas[k]
        for op, c in zip(ops, coeffs):
            wires = np.where(np.asarray(op) == 1)[0]
            if len(wires) == 0:
                continue  
            qp.MultiRZ(2.0 * g * c, wires=[int(w) for w in wires])
        b = betas[k]
        for w in range(n_qubits):
            qp.RX(2.0 * b, wires=w)


def n_params_qaoa(p: int) -> int:
    return 2 * p


def n_params_for(ansatz: str, spec, n_qubits: int) -> int:
    if ansatz == "iqp":
        if spec == "single":
            return n_qubits
        if spec == "circular":
            return 2 * n_qubits
        if spec == "full":
            return n_qubits + n_qubits * (n_qubits - 1) // 2
        raise ValueError(f"IQP mode không hợp lệ: {spec!r}")
    if ansatz == "hea":
        return n_params_hea(n_qubits, spec)
    if ansatz == "qaoa":
        return n_params_qaoa(spec)
    raise ValueError(f"ansatz không hợp lệ: {ansatz!r}")


def apply_ansatz(params, ops, coeffs, n_qubits: int, ansatz: str, spec) -> None:
    if ansatz == "iqp":
        iqp_layer(params[:n_qubits], params[n_qubits:], spec)
    elif ansatz == "hea":
        hea_layer(params, n_qubits, spec)
    elif ansatz == "qaoa":
        qaoa_layer(params, ops, coeffs, n_qubits, spec)
    else:
        raise ValueError(f"ansatz không hợp lệ: {ansatz!r}")


PI4_EPS = 0.05  


def init_params(name: str, n_params: int, n_qubits: int):
    if name == "normal":
        return np.random.normal(0.0, 1.0, n_params)
    if name == "uniform":
        return np.random.uniform(-np.pi, np.pi, n_params)
    if name == "pi4":
        return np.random.uniform(np.pi / 4 - PI4_EPS, np.pi / 4 + PI4_EPS, n_params)
    if name == "he":
        return np.random.normal(0.0, np.sqrt(2.0 / n_qubits), n_params)
    if name == "glorot":
        return np.random.normal(0.0, np.sqrt(1.0 / n_qubits), n_params)
    raise ValueError(f"init không hợp lệ: {name!r}")
