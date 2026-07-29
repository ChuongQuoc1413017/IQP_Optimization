import pennylane as qp
import numpy as np

def number_partition_obs(numbers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the observables and coefficients corresponding to the
    Number Partitioning cost Hamiltonian.

    The Number Partitioning Hamiltonian is

        H = Σ_{i,j} numbers[i] * numbers[j] * O_{ij}

    where

        O_{ij} = I               if i == j
        O_{ij} = Z_i Z_j         if i != j

    In IQPopt, observables are represented as binary vectors:
    - 0 indicates the identity operator on a qubit.
    - 1 indicates a Pauli-Z operator on a qubit.

    For example, for 4 qubits:
        [1, 0, 1, 0]
    represents the observable Z_0 x Z_2.

    The identity observable is represented by:
        [0, 0, 0, 0]

    Parameters
    ----------
    numbers : list[int]
        List of integers defining the number partitioning instance.
        The length of the list determines the number of qubits.

    Returns
    -------
    ops : np.ndarray
        Array of shape (n_qubits^2, n_qubits) containing the observables.
        Each row is a binary vector encoding either an identity operator
        or a two-qubit Pauli-Z correlation.

    coeffs : np.ndarray
        Array of shape (n_qubits^2,) containing the coefficient associated
        with each observable. The k-th coefficient corresponds to the
        k-th observable in `ops`.
    """

    n_qubits = len(numbers)

    ops = []
    coeffs = []

    for i in range(n_qubits):
        for j in range(n_qubits):

            # Coefficient of O_{ij}
            coeffs.append(numbers[i] * numbers[j])

            # Binary representation of the observable
            ob = [0] * n_qubits

            # For i != j, encode Z_i Z_j
            if i != j:
                ob[i] = 1
                ob[j] = 1

            # For i == j, keep all zeros to represent Identity
            ops.append(ob)

    return np.array(ops), np.array(coeffs)

def maxcut_obs(
        n_qubits: int, 
        edges: list[tuple[int, int]], 
        weights: list[float] | None = None, 
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct observables and coefficients for the MaxCut Hamiltonian

        H = - Σ_{i,j in E} w_{ij} * (I - Z_i Z_j)/2

    Parameters
    ----------
    n_qubits : int
        Number of vertices/qubits.

    edges : list[(int, int)]
        List of graph edges.

    weights : list[float] | None
        Edge weights. If None, all weights are taken as 1.

    Returns
    -------
    ops : np.ndarray
        Binary observable encodings.

    coeffs : np.ndarray
        Corresponding coefficients.
    """

    if weights is None:
        weights = [1.0] * len(edges)

    ops = [[0] * n_qubits]

    # Identity term: + w/2
    coeffs = [sum(weights)/2]

    for (i, j), w in zip(edges, weights):
        # ZZ term: - w/2 * Z_i Z_j
        ob = [0] * n_qubits
        ob[i] = 1
        ob[j] = 1

        ops.append(ob)
        coeffs.append(-w/2)

    return np.array(ops), -np.array(coeffs)

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

def ising_obs(n_qubits, corr=2):
    obs = []
    for i in range(n_qubits):
        v = [0] * n_qubits; v[i] = 1; obs.append(v)
    if corr == 2:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                v = [0] * n_qubits; v[i] = 1; v[j] = 1; obs.append(v)
    return np.array(obs)

######################################################################
####### EXACT GROUND TRUTH 
######################################################################
def maxcut_value(bitstring: list[int], 
                 edges: list[tuple[int, int]], 
                 weights: list[float], 
                ) -> float:
    """
    Compute the MaxCut objective value for a given partition.

    Parameters
    ----------
    bitstring : list[int]
        Binary partition assignment.
        0 and 1 represent the two sides of the cut.

    edges : list[(int, int)]
        Edge list.

    weights : list[float]
        Edge weights.

    Returns
    -------
    float
        Cut value.
    """

    cut_value = 0.0

    for (u, v), w in zip(edges, weights):
        if bitstring[u] != bitstring[v]:
            cut_value += w

    return cut_value

def solve_maxcut_exact_symmetry(
    edges: list[tuple[int, int]],
    weights: list[float],
):
    n = max(max(u, v) for u, v in edges) + 1

    best_cut = -np.inf
    best_partition = None

    for mask in range(1 << (n - 1)):

        cut_value = 0.0

        for (u, v), w in zip(edges, weights):

            side_u = 0 if u == 0 else ((mask >> (u - 1)) & 1)
            side_v = 0 if v == 0 else ((mask >> (v - 1)) & 1)

            if side_u != side_v:
                cut_value += w

        if cut_value > best_cut:
            best_cut = cut_value

            partition = np.zeros(n, dtype=int)
            for i in range(1, n):
                partition[i] = (mask >> (i - 1)) & 1

            best_partition = partition

    return best_cut, best_partition

def sum_selected(binary_str: str, numbers: list[int]) -> int:
    """
    Compute the sum of selected elements in `numbers` based on a binary mask.

    Each character in `binary_str` acts as a selector:
    - '1' includes the corresponding element in `numbers`
    - '0' excludes it

    Parameters
    ----------
    binary_str : str
        A string of '0' and '1' characters representing a selection mask.

    numbers : list[int]
        List of integers to be filtered and summed.

    Returns
    -------
    int
        Sum of elements in `numbers` where the corresponding bit in
        `binary_str` is '1'.

    Raises
    ------
    ValueError
        If the length of `binary_str` does not match the length of `numbers`.

    Examples
    --------
    >>> sum_selected("101", [3, 5, 7])
    10
    """
    
    binary_arr = np.array(list(binary_str)) == '1'
    numbers_arr = np.array(numbers)

    if len(binary_arr) != len(numbers_arr):
        raise ValueError("Binary string and numbers list must have the same length")

    return numbers_arr[binary_arr].sum()