import iqpopt as iqp
import pennylane as qp
import numpy as np
import jax
import jax.numpy as jnp
# from pennylane.qnn import iqp_expval as op_expval # remove

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

def gens(n_qubits: int, mode: str = "full") -> list:
    """
    Generate the list of IQP generator gates for an IQP circuit.

    Each generator is represented as a list containing a single term:
    - [[i]] represents a one-qubit Z rotation generator on qubit i.
    - [[i, j]] represents a two-qubit ZZ interaction generator between
      qubits i and j.

    The generated set always includes all single-qubit generators.
    Additional two-qubit generators are determined by the selected mode.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the IQP circuit.

    mode : str, optional
        Connectivity pattern for the two-qubit generators.

        Supported values are:

        - "full":
          Add a ZZ generator between every pair of qubits.
          This produces a fully connected interaction graph.

        - "circular":
          Add ZZ generators only between nearest neighbours in a ring.
          Qubit i is connected to i+1, and the last qubit is connected
          back to the first.

        Default is "full".

    Returns
    -------
    list
        List of IQP generators.

    Notes
    -----
    The total number of generators is:

    - mode="full":
      n_qubits + n_qubits * (n_qubits - 1) / 2

    - mode="circular":
      2 * n_qubits

    These generators can be passed directly to
    ``iqp.IqpSimulator(n_qubits, gates)``.
    """

    gates = []

    # Add all single-qubit generators.
    for i in range(n_qubits):
        gates.append([[i]])

    # Add all pairwise ZZ generators.
    if mode == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                gates.append([[i, j]])

    # Add nearest-neighbour ZZ generators on a ring.
    if mode == "circular":
        for i in range(n_qubits - 1):
            gates.append([[i, i + 1]])

        # Close the ring.
        gates.append([[n_qubits - 1, 0]])

    # Add all triplewise ZZZ generators.
    if mode == "triple":
        
        # Add ZZ
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                gates.append([[i, j]])
                
        ## Add ZZZ
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(j + 1, n_qubits):
                    gates.append([[i, j, k]])

    return gates

def loss_fn(
        params: jnp.ndarray,
        circuit,
        ops: np.ndarray,
        coeffs: np.ndarray,
        n_samples: int,
        key: jax.Array,
    ) -> jnp.ndarray:
    """
    Compute the expectation value of the cost Hamiltonian.

    The cost function is evaluated as

        C(θ) = Σ_i coeffs[i] . ⟨O_i⟩

    where:
    - O_i are the observables specified in `ops`,
    - ⟨O_i⟩ are their expectation values under the IQP circuit
      parameterized by `params`,
    - coeffs[i] are the corresponding Hamiltonian coefficients.

    This function can be used directly with the IQPopt optimizer to
    variationally minimize the Hamiltonian objective.

    Parameters
    ----------
    params : jnp.ndarray
        Array of circuit parameters. The length must match the number
        of generators in the IQP circuit.

    circuit : iqp.IqpSimulator
        IQP circuit simulator containing the circuit structure and
        generator definitions.

    ops : np.ndarray
        Array of observables.
        Shape is (n_observables, n_qubits), where each row is a binary
        encoding of a Pauli-Z string.

    coeffs : np.ndarray
        Array of Hamiltonian coefficients associated with the
        observables in `ops`.
        Shape is (n_observables,).

    n_samples : int
        Number of Monte Carlo samples used to estimate expectation
        values.

    key : jax.Array
        JAX pseudo-random number generator key used for sampling.

    Returns
    -------
    jnp.ndarray
        Scalar expectation value of the Hamiltonian.
        Lower values correspond to better solutions.

    Notes
    -----
    The expectation values are computed using IQPopt's fast estimator:

        expvals = op_expval(...)[0]

    and combined into the Hamiltonian expectation value through a
    weighted sum:

        C = coeffs . expvals

    which is equivalent to evaluating the expectation value of the
    corresponding PennyLane Hamiltonian.
    """

    # Estimate expectation values of all observables.
    expvals = op_expval(
        ops,
        params,
        circuit.gates,
        circuit.n_qubits,
        n_samples,
        key,
    )[0]

    # Compute the Hamiltonian expectation value.
    return jnp.dot(expvals, coeffs)

def train_model(
        circuit,
        ops: np.ndarray,
        coeffs: np.ndarray,
        params_init: np.ndarray,
        key: jax.Array,
        loss_fn,
        optimizer: str = "Adam",
        stepsize: float = 1e-3,
        n_iters: int = 4000,
        n_samples: int = 10000,
    ) -> iqp.Trainer:
    """
    Train an IQP circuit to minimize the cost function.

    Parameters
    ----------
    circuit : iqp.IqpSimulator
        IQP circuit simulator containing the circuit structure and
        generator definitions.

    ops : np.ndarray
        Array of observables defining the Hamiltonian.

    coeffs : np.ndarray
        Coefficients associated with the observables in `ops`.

    params_init : np.ndarray
        Initial parameters for the IQP circuit optimization.
        Must have length equal to `len(circuit.gates)`.

    key : jax.Array
        JAX pseudo-random number generator key used during expectation
        value estimation.

    loss_fn : callable
        Cost function used for optimization.

    optimizer : str, optional
        Name of the optimizer used by IQPopt.
        Default is "Adam".

    stepsize : float, optional
        Learning rate of the optimizer.
        Default is 0.001.

    n_iters : int, optional
        Number of optimization iterations.
        Default is 4000.

    n_samples : int, optional
        Number of samples used to estimate expectation values during
        training.
        Default is 10000.

    Returns
    -------
    iqp.Trainer
        Trained IQPopt trainer object containing the optimization
        results, including the final parameters in
        `trainer.final_params`.
    """

    trainer = iqp.Trainer(optimizer, loss_fn, stepsize)

    loss_kwargs = {
        "params": params_init,
        "circuit": circuit,
        "ops": ops,
        "coeffs": coeffs,
        "n_samples": n_samples,
        "key": key,
    }

    trainer.train(n_iters, loss_kwargs)

    return trainer