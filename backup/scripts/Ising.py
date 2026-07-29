import numpy as np

def exact_ising_solution(
    observables: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Find the exact ground state of an Ising Hamiltonian by exhaustive search.

    The Hamiltonian is assumed to be

        H = Σ_i c_i O_i,

    where each observable O_i is encoded as a binary vector. A value of
    ``1`` indicates a Z operator acting on that qubit and ``0`` indicates
    the identity.

    Parameters
    ----------
    observables : np.ndarray
        Array of shape ``(n_observables, n_qubits)``. Each row specifies
        one Ising observable.

    coefficients : np.ndarray
        Coefficient for each observable.

    Returns
    -------
    ground_state : np.ndarray
        Spin configuration (±1) minimizing the Hamiltonian.

    ground_energy : float
        Minimum energy.
    """

    n_qubits = observables.shape[1]

    ground_energy = np.inf
    ground_state = None

    # Iterate over all spin configurations
    for spins in itertools.product([-1, 1], repeat=n_qubits):
        spins = np.array(spins)

        energy = 0.0

        for obs, coeff in zip(observables, coefficients):
            support = np.where(obs == 1)[0]
            value = np.prod(spins[support]) if len(support) else 1.0
            energy += coeff * value

        if energy < ground_energy:
            ground_energy = energy
            ground_state = spins.copy()

    return ground_state, ground_energy