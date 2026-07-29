import pennylane as qp
import numpy as np
from pennylane import numpy as qnp
from IQP_circuit import *

def binary_ops_to_hamiltonian(ops: list[np.ndarray], coeffs: np.ndarray) -> qp.Hamiltonian:
    """Converts a binary representation of Pauli-Z operators into a PennyLane Hamiltonian.

    Each binary operator is interpreted as a tensor product of Pauli-Z operators,
    where a value of ``1`` indicates the presence of a Pauli-Z on the corresponding
    qubit and ``0`` indicates the identity. An all-zero bitstring is mapped to the
    identity observable.

    Args:
        ops (list[np.ndarray]): List of binary vectors representing Pauli-Z operators.
        coeffs (np.ndarray): Coefficients associated with each operator.

    Returns:
        qp.Hamiltonian: PennyLane Hamiltonian constructed from the given
        coefficients and observables.
    """
    terms = []

    # Convert each binary operator into its corresponding PennyLane observable.
    for op in ops:
        # Identify the qubits on which a Pauli-Z acts.
        wires = np.where(np.asarray(op) == 1)[0]

        # If no Pauli-Z acts on any qubit, use the identity observable.
        if len(wires) == 0:
            observable = qp.Identity(wires = 0)

        # A single active qubit corresponds to a single Pauli-Z operator.
        elif len(wires) == 1:
            observable = qp.PauliZ(wires[0])

        # Multiple active qubits correspond to a tensor product of Pauli-Z operators.
        else:
            observable = qp.prod(*[qp.PauliZ(w) for w in wires])

        terms.append(observable)

    return qp.Hamiltonian(coeffs, terms)

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
    
    # Apply the initial Hadamard layer.
    for i in range(n_qubits):
        qp.Hadamard(wires = i)

    # Apply single-qubit Z rotations.
    for i in range(n_qubits):
        qp.RZ(single_params[i], wires = i)

    # Apply two-qubit ZZ interaction gates according to the selected mode.
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

    # Apply the final Hadamard layer.
    for i in range(n_qubits):
        qp.Hadamard(wires = i)

def iqp_circuit(
    single_params: np.ndarray,
    pair_params: np.ndarray,
    H: qp.Hamiltonian,
    mode_circuit: str = "single",
) -> qp.measurements.ExpectationMP:
    """Defines the IQP circuit that computes the expectation value of a Hamiltonian.

    The circuit consists of a single IQP layer followed by the measurement
    of the expectation value of the input Hamiltonian.

    Args:
        single_params (np.ndarray): Rotation angles for the single-qubit
            RZ gates. Shape ``(n_qubits,)``.
        pair_params (np.ndarray): Rotation angles for the two-qubit
            IsingZZ gates. The required shape depends on ``mode``.
        H (qp.Hamiltonian): Hamiltonian whose expectation value is
            measured.
        mode_circuit (str, optional): Connectivity pattern of the IQP layer.
            Supported options are ``"single"``, ``"circular"``, and
            ``"full"``. Defaults to ``"single"``.

    Returns:
        qp.measurements.ExpectationMP: PennyLane circuit with an
        expectation value measurement.
    """
    # Apply a single IQP layer.
    iqp_layer(
        single_params = single_params,
        pair_params = pair_params,
        mode_circuit = mode_circuit,
    )

    # Measure the expectation value of the Hamiltonian.
    return qp.expval(H)

def cost(
    single_params: np.ndarray,
    pair_params: np.ndarray,
    H: qp.Hamiltonian,
    mode_circuit: str = "single",
) -> float:
    """Evaluates the expectation value of a Hamiltonian using an IQP circuit.

    This function constructs a PennyLane quantum device, instantiates a
    QNode for the IQP circuit, and returns the expectation value of the
    input Hamiltonian.

    Args:
        single_params (np.ndarray): Rotation angles for the single-qubit
            RZ gates. Shape ``(n_qubits,)``.
        pair_params (np.ndarray): Rotation angles for the two-qubit
            IsingZZ gates. The required shape depends on ``mode``.
        H (qp.Hamiltonian): Hamiltonian whose expectation value is
            measured.
        mode_circuit (str, optional): Connectivity pattern of the IQP layer.
            Supported options are ``"single"``, ``"circular"``, and
            ``"full"``. Defaults to ``"single"``.

    Returns:
        float: Expectation value of the Hamiltonian.
    """
    # Create the quantum device.
    dev = qp.device("lightning.qubit", wires = len(single_params), shots = None)

    # Construct the QNode for the IQP circuit.
    qnode = qp.QNode(iqp_circuit, dev) 

    # Evaluate the expectation value of the Hamiltonian.
    return qnode(single_params, pair_params, H, mode_circuit)

def hamiltonian_data(
    data: list[np.ndarray],
    n_qubits: int,
    mode_ham: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the Hamiltonian representation for different problem modes.

    This function converts structured input data into:
    1. A list of binary Pauli-Z observables (ops),
    2. A corresponding coefficient vector (coeffs).

    The output depends on the chosen Hamiltonian mode:
    - "ising": constructs Ising-type observables and uses provided coefficients.
    - "maxcut": converts graph edges and weights into a MaxCut Hamiltonian.
    - "partition": constructs a number partitioning Hamiltonian.

    Args:
        data (list[np.ndarray]): Input data defining the problem instance.
        n_qubits (int): Number of qubits in the system.
        mode_ham (str): Specifies the Hamiltonian type ("ising", "maxcut", "partition").

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Binary observable representation (ops)
            - Coefficients associated with each observable
    """

    # Select Hamiltonian construction based on the chosen mode.
    if mode_ham == "ising":
        coeffs = np.asarray(data)
        ops = ising_obs(n_qubits, 2)

    elif mode_ham == "maxcut":
        edges, weights = data[0], data[1]
        ops, coeffs = maxcut_obs(n_qubits, edges, weights)

    elif mode_ham == "partition":
        numbers = data
        ops, coeffs = number_partition_obs(numbers)

    else:
        raise ValueError(f"Unknown mode_ham: {mode_ham}")

    # Generate Hamiltonian representation.
    return ops, coeffs

def prepare_params(
    mode_circuit: str,
    n_qubits: int,
    theta: float,
    epsilon: float,
) -> tuple[qnp.ndarray, qnp.ndarray]:
    """Initializes trainable parameters for an IQP circuit.

    Depending on the selected circuit mode, this function determines the
    number of two-qubit interaction parameters and samples both single-
    qubit and pairwise parameters uniformly around ``theta``.

    Args:
        mode_circuit (str): Connectivity pattern of the IQP layer. Supported
            options are ``"single"``, ``"circular"``, and ``"full"``.
        n_qubits (int): Number of qubits in the circuit.
        theta (float): Central value around which parameters are initialized.
        epsilon (float): Uniform sampling range around ``theta``.

    Returns:
        tuple[qnp.ndarray, qnp.ndarray]:
            - single_params: Shape ``(n_qubits,)``
            - pair_params: Shape depends on ``mode``:
                * ``"single"``: 0
                * ``"circular"``: ``(n_qubits,)``
                * ``"full"``: ``(n_qubits * (n_qubits - 1) // 2,)``
    """
    # Determine number of pairwise interaction parameters.
    if mode_circuit == "single":
        n_pairs = 0
    elif mode_circuit == "circular":
        n_pairs = n_qubits
    elif mode_circuit == "full":
        n_pairs = n_qubits * (n_qubits - 1) // 2
    else:
        raise ValueError(
            f"Unsupported mode '{mode_circuit}'. "
            "Expected one of {'single', 'circular', 'full'}."
        )
        
    # Initialize single-qubit parameters.
    single_params = qnp.random.uniform(
        theta - epsilon,
        theta + epsilon,
        size=n_qubits,
        requires_grad=True,
    )

    # Initialize two-qubit parameters.
    pair_params = qnp.random.uniform(
        theta - epsilon,
        theta + epsilon,
        size=n_pairs,
        requires_grad=True,
    )

    expected = {
        "single": 0,
        "circular": n_qubits,
        "full": n_qubits * (n_qubits - 1) // 2,
    }[mode_circuit]
    if len(pair_params) != expected:
        raise ValueError("Expected one of {'single', 'circular', 'full'}.")

    return single_params, pair_params

def n_pair_params(mode_circuit: str, n_qubits: int) -> int:
    """Number of two-qubit (IsingZZ) parameters for a given connectivity.

    - ``"single"``   -> 0
    - ``"circular"`` -> n_qubits (nearest-neighbour ring)
    - ``"full"``     -> n_qubits * (n_qubits - 1) // 2 (all-to-all)
    """
    if mode_circuit == "single":
        return 0
    if mode_circuit == "circular":
        return n_qubits
    if mode_circuit == "full":
        return n_qubits * (n_qubits - 1) // 2
    raise ValueError(
        f"Unsupported mode '{mode_circuit}'. "
        "Expected one of {'single', 'circular', 'full'}."
    )


def prepare_params_normal(
    mode_circuit: str,
    n_qubits: int,
) -> tuple[qnp.ndarray, qnp.ndarray]:
    """Standard Gaussian initialization for IQP circuit parameters.

    Both single-qubit (RZ) and pairwise (IsingZZ) angles are sampled
    independently from a standard normal distribution::

        theta ~ N(0, 1)

    Args:
        mode_circuit (str): One of ``"single"``, ``"circular"``, ``"full"``.
        n_qubits (int): Number of qubits.

    Returns:
        tuple[qnp.ndarray, qnp.ndarray]: ``(single_params, pair_params)``.
    """
    n_pairs = n_pair_params(mode_circuit, n_qubits)

    single_params = qnp.random.normal(0.0, 1.0, size=n_qubits, requires_grad=True)
    pair_params = qnp.random.normal(0.0, 1.0, size=n_pairs, requires_grad=True)

    return single_params, pair_params


def prepare_params_uniform(
    mode_circuit: str,
    n_qubits: int,
) -> tuple[qnp.ndarray, qnp.ndarray]:
    """Uniform initialization for IQP circuit parameters.

    Both single-qubit (RZ) and pairwise (IsingZZ) angles are sampled
    independently from a uniform distribution over the full angle range::

        theta ~ U(-pi, pi)

    Args:
        mode_circuit (str): One of ``"single"``, ``"circular"``, ``"full"``.
        n_qubits (int): Number of qubits.

    Returns:
        tuple[qnp.ndarray, qnp.ndarray]: ``(single_params, pair_params)``.
    """
    n_pairs = n_pair_params(mode_circuit, n_qubits)

    single_params = qnp.random.uniform(-np.pi, np.pi, size=n_qubits, requires_grad=True)
    pair_params = qnp.random.uniform(-np.pi, np.pi, size=n_pairs, requires_grad=True)

    return single_params, pair_params


def prepare_params_pi4(
    mode_circuit: str,
    n_qubits: int,
    epsilon: float = 0.05,
) -> tuple[qnp.ndarray, qnp.ndarray]:
    """Near-``pi/4`` initialization for IQP circuit parameters.

    Both single-qubit (RZ) and pairwise (IsingZZ) angles are sampled
    uniformly in a small window centred on ``pi/4``::

        theta ~ U(pi/4 - epsilon, pi/4 + epsilon)

    Thin wrapper around :func:`prepare_params` with ``theta = pi/4``.

    Args:
        mode_circuit (str): One of ``"single"``, ``"circular"``, ``"full"``.
        n_qubits (int): Number of qubits.
        epsilon (float, optional): Half-width of the uniform window around
            ``pi/4``. Defaults to ``0.05``.

    Returns:
        tuple[qnp.ndarray, qnp.ndarray]: ``(single_params, pair_params)``.
    """
    return prepare_params(mode_circuit, n_qubits, np.pi / 4, epsilon)


def prepare_params_glorot(
    mode_circuit: str,
    n_qubits: int,
) -> tuple[qnp.ndarray, qnp.ndarray]:

    n_pairs = n_pair_params(mode_circuit, n_qubits)

    std = np.sqrt(1.0 / n_qubits)
    single_params = qnp.random.normal(0.0, std, size=n_qubits, requires_grad=True)
    pair_params = qnp.random.normal(0.0, std, size=n_pairs, requires_grad=True)

    return single_params, pair_params


def prepare_params_he(
    mode_circuit: str,
    n_qubits: int,
) -> tuple[qnp.ndarray, qnp.ndarray]:

    n_pairs = n_pair_params(mode_circuit, n_qubits)

    std = np.sqrt(2.0 / n_qubits)
    single_params = qnp.random.normal(0.0, std, size=n_qubits, requires_grad=True)
    pair_params = qnp.random.normal(0.0, std, size=n_pairs, requires_grad=True)

    return single_params, pair_params


def circuit_probs(
    single_params: np.ndarray,
    pair_params: np.ndarray,
    mode_circuit: str = "single",
) -> qp.measurements.ProbabilityMP:
    """Defines the IQP circuit that computes the output state probabilities.

    The circuit consists of a single IQP layer followed by a measurement
    of the computational basis probability distribution.

    Args:
        single_params (np.ndarray): Rotation angles for the single-qubit
            RZ gates. Shape ``(n_qubits,)``.
        pair_params (np.ndarray): Rotation angles for the two-qubit
            IsingZZ gates. The required shape depends on ``mode``.
        mode_circuit (str, optional): Connectivity pattern of the IQP layer.
            Supported options are ``"single"``, ``"circular"``, and
            ``"full"``. Defaults to ``"single"``.

    Returns:
        qp.measurements.ProbabilityMP: PennyLane circuit with a
        probability measurement over all computational basis states.
    """
    # Apply a single IQP layer.
    iqp_layer(
        single_params = single_params,
        pair_params = pair_params,
        mode_circuit = mode_circuit,
    )

    # Measure the computational basis probabilities.
    n_qubits = len(single_params)
    return qp.probs(wires = range(n_qubits))


def state_probs(
    single_params: np.ndarray,
    pair_params: np.ndarray,
    mode_circuit: str = "single",
) -> np.ndarray:
    """Evaluates the output state probabilities of an IQP circuit.

    This function constructs a PennyLane quantum device, instantiates a
    QNode for the IQP circuit, and returns the probability distribution
    over all computational basis states.

    Args:
        single_params (np.ndarray): Rotation angles for the single-qubit
            RZ gates. Shape ``(n_qubits,)``.
        pair_params (np.ndarray): Rotation angles for the two-qubit
            IsingZZ gates. The required shape depends on ``mode``.
        mode_circuit (str, optional): Connectivity pattern of the IQP layer.
            Supported options are ``"single"``, ``"circular"``, and
            ``"full"``. Defaults to ``"single"``.

    Returns:
        np.ndarray: Probability vector of length ``2**n_qubits``
        corresponding to the computational basis states.
    """
    # Create the quantum device.
    dev = qp.device("lightning.qubit", wires=len(single_params))

    # Construct the QNode for the probability circuit.
    qnode = qp.QNode(circuit_probs, dev)

    # Evaluate the computational basis probabilities.
    return qnode(
        single_params,
        pair_params,
        mode_circuit,
    )