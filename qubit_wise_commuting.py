from typing import List
import numpy as np
import rustworkx as rx
from collections import defaultdict

from openfermion import count_qubits, QubitOperator


def create_pauli_magical_mat_from_openfermion(operator: QubitOperator):
    n_qubit = count_qubits(operator)

    id_dict = {"I": 0, "X": 2, "Y": 3, "Z": 1}
    mat = np.zeros((len(operator.terms), n_qubit))
    for i, term in enumerate(operator.terms):
        for j, pauli_id in term:
            mat[i, j] = id_dict[pauli_id]
    return mat[:, ::-1]


def noncommutation_graph(operator: QubitOperator):
    """Create an edge list representing the non-commutation graph (Pauli Graph).

    An edge (i, j) is present if i and j are not commutable.

    Returns:
        list[tuple[int,int]]: A list of pairs of indices of the PauliList that are not commutable.
    """
    # convert a Pauli operator into int vector where {I: 0, X: 2, Y: 3, Z: 1}
    mat1 = create_pauli_magical_mat_from_openfermion(operator)
    mat2 = mat1[:, None]
    # This is 0 (false-y) iff one of the operators is the identity and/or both operators are the
    # same.  In other cases, it is non-zero (truth-y).
    qubit_anticommutation_mat = (mat1 * mat2) * (mat1 - mat2)
    adjacency_mat = np.logical_or.reduce(qubit_anticommutation_mat, axis=2)

    return list(zip(*np.where(np.triu(adjacency_mat, k=1))))


def create_graph(operator: QubitOperator):
    """Transform measurement operator grouping problem into graph coloring problem

    Returns:
        rustworkx.PyGraph: A class of undirected graphs
    """

    edges = noncommutation_graph(operator)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(len(operator.terms)))
    graph.add_edges_from_no_data(edges)
    return graph


def group_commuting(operator: QubitOperator) -> QubitOperator:
    """Partition a PauliList into sets of qubit-wise commuting Pauli strings.

    Returns:
        list[PauliList]: List of PauliLists where each PauliList contains commuting Pauli operators.
    """

    graph = create_graph(operator)
    # Keys in coloring_dict are nodes, values are colors
    coloring_dict = rx.graph_greedy_color(graph)
    groups = defaultdict(list)
    for idx, color in coloring_dict.items():
        groups[color].append(idx)
    ops_list = np.array(list(operator.terms.keys()), dtype=object)
    groups_mapped = [[ops_list[idx] for idx in group] for group in groups.values()]
    groups_unpacked = [set([g for subgroupg in group for g in subgroupg]) for group in groups_mapped]
    return sum([QubitOperator(list(group)) / len(groups) for group in groups_unpacked])
