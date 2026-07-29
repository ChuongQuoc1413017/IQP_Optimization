import numpy as np
import random
from itertools import combinations

def count_partitions(nums: list[int]) -> int:
    """
    Count the number of unique ways to partition `nums` into two subsets
    with equal sum using brute-force enumeration.

    Each partition is counted only once (A|B is considered the same as B|A).

    Parameters
    ----------
    nums : list[int]
        List of integers to partition.

    Returns
    -------
    int
        Number of distinct equal-sum partitions.
        Returns 0 if total sum is odd.
    """

    total = sum(nums)

    # Equal partition impossible if total is odd
    if total % 2 != 0:
        return 0

    target = total // 2
    n = len(nums)

    solutions = []

    for r in range(1, n):
        for subset in combinations(range(n), r):
            subset_sum = sum(nums[i] for i in subset)

            if subset_sum == target:
                other = tuple(sorted(set(range(n)) - set(subset)))

                # Canonical representation to avoid counting
                # A|B and B|A separately
                partition = tuple(sorted([
                    tuple(sorted(subset)),
                    other
                ]))

                if partition not in solutions:
                    solutions.append(partition)

    return len(solutions)

def generate_unique_partition_instance(n: int = 8, 
                                       min_value: int = 1, 
                                       max_value: int = 30, 
                                       max_attempts: int = 100000) -> list[int] | None:
    """
    Generate a number partition instance with exactly one valid equal-sum partition.

    The function randomly samples integer lists and keeps only those
    for which `count_partitions(nums) == 1`.

    Parameters
    ----------
    n : int
        Number of elements in the instance.

    min_value : int
        Minimum value of each integer.

    max_value : int
        Maximum value of each integer.

    max_attempts : int
        Maximum number of random samples to try before giving up.

    Returns
    -------
    list[int] | None
        A list of integers that admits exactly one equal-sum partition.
        Returns None if no such instance is found within `max_attempts`.

    Notes
    -----
    - This is a stochastic brute-force generator.
    - Runtime may be extremely slow for large `n`.
    """

    for _ in range(max_attempts):
        nums = [random.randint(min_value, max_value) for _ in range(n)]

        if count_partitions(nums) == 1:
            return nums

    return None

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