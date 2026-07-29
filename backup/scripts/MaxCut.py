import numpy as np

def random_maxcut_instance(n_qubits: int,
                           p: float = 0.5,
                           weighted: bool = True,
                           seed: int | None = None, 
                          ) -> tuple[list[(int, int)], list[float]]:
    """
    Generate a random graph suitable for maxcut_obs().

    Parameters
    ----------
    n_qubits : int
        Number of vertices.

    p : float
        Edge probability for Erdős–Rényi G(n,p).

    weighted : bool
        If True, assign random integer weights in [1,10].

    seed : int | None
        Random seed.

    Returns
    -------
    edges : list[(int, int)]
    weights : list[float]
    """
    
    rng = np.random.default_rng(seed)

    edges = []
    weights = []

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):

            if rng.random() < p:
                edges.append((i, j))

                if weighted:
                    weights.append(float(rng.integers(1, 11)))
                else:
                    weights.append(1.0)

    return edges, weights

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