import numpy as np
import pennylane as qp
from pennylane import numpy as qnp

def iqp_layer(
    single_params: np.ndarray,
    pair_params: np.ndarray,
    mode_circuit: str = "single",
) -> None:
    """Applies a single Instantaneous Quantum Polynomial (IQP) layer.

    The layer consists of an initial Hadamard transform, followed by
    commuting diagonal gates. Depending on ``mode``, the diagonal gates
    include only single-qubit Z rotations, nearest-neighbor ZZ
    interactions on a ring, or all-to-all ZZ interactions. The layer
    concludes with a final Hadamard transform.

    Args:
        single_params (np.ndarray): Rotation angles for the single-qubit
            RZ gates. Shape ``(n_qubits,)``.
        pair_params (np.ndarray): Rotation angles for the two-qubit
            IsingZZ gates. The required shape depends on ``mode``:
            - ``"single"``: not used.
            - ``"circular"``: ``(n_qubits,)``.
            - ``"full"``: ``(n_qubits * (n_qubits - 1) // 2,)``.
        mode_circuit (str, optional): Connectivity pattern of the IQP layer.
            Supported options are:
            - ``"single"``: only single-qubit Z rotations.
            - ``"circular"``: nearest-neighbor ZZ interactions with
              periodic boundary conditions.
            - ``"full"``: all-to-all ZZ interactions.
            Defaults to ``"single"``.

    Returns:
        None: The function applies quantum operations directly to the
        current PennyLane quantum tape.
    """
    n_qubits = len(single_params)
    
    for i in range(n_qubits):
        qp.Hadamard(wires = i)

    for i in range(n_qubits):
        qp.RZ(single_params[i], wires = i)

    if mode_circuit == "circular":
        pair_idx = 0
        for i in range(n_qubits - 1):
            qp.IsingZZ(pair_params[pair_idx], wires = [i, i + 1])
            pair_idx += 1
        qp.IsingZZ(pair_params[pair_idx], wires = [n_qubits - 1, 0])
        pair_idx += 1

    elif mode_circuit == "full":
        pair_idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qp.IsingZZ(pair_params[pair_idx], wires = [i, j])
                pair_idx += 1

    elif mode_circuit != "single":
        raise ValueError(
            f"Unsupported mode '{mode_circuit}'. "
            "Expected one of {'single', 'circular', 'full'}."
        )

    for i in range(n_qubits):
        qp.Hadamard(wires = i)

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
        raise ValueError(f"Invalid IQP mode: {spec!r}")
    if ansatz == "hea":
        return n_params_hea(n_qubits, spec)
    if ansatz == "qaoa":
        return n_params_qaoa(spec)
    raise ValueError(f"Invalid ansatz: {ansatz!r}")

def apply_iqp_layers(params, n_qubits: int, mode: str, n_layers: int) -> None:
    per = n_params_for("iqp", mode, n_qubits)
    for l in range(n_layers):
        seg = params[l * per:(l + 1) * per]
        iqp_layer(seg[:n_qubits], seg[n_qubits:], mode)


def apply_ansatz(params, ops, coeffs, n_qubits: int, ansatz: str, spec) -> None:
    if ansatz == "iqp":
        iqp_layer(params[:n_qubits], params[n_qubits:], spec)
    elif ansatz == "hea":
        hea_layer(params, n_qubits, spec)
    elif ansatz == "qaoa":
        qaoa_layer(params, ops, coeffs, n_qubits, spec)
    else:
        raise ValueError(f"Invalid ansatz: {ansatz!r}")

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
    raise ValueError(f"Invalid initialization: {name!r}")
