import numpy as np

from openfermion import QubitOperator

from typing import List
from utils import create_pauli_id_from_openfermion, pad_op


def derandomized_classical_shadow(operators: QubitOperator, meas_cnt: int, n_qubit: int):
    epsilon = 0.95
    nu = 1 - np.exp(-(epsilon**2) / 2)
    operators_in = pad_op(operators, n_qubit)
    # 0:'I', 1:'X', 2:'Y', 3:'Z'
    pauli_ids = np.array([create_pauli_id_from_openfermion(op, n_qubit) for op in operators_in.terms])
    num_terms = len(operators_in.terms)

    meas_axes = [[] for i in range(meas_cnt)]

    def hit(a, b):
        return (a == 0) | (a == b)

    def weight(a):
        return (a != 0).sum(axis=1)

    all_hit_cnt = np.zeros(num_terms)
    history_cnt = 0
    all_hit_memo = np.ones(num_terms, dtype=bool)

    def calc_expected_conf(W: int, m: int, k: int):
        nonlocal all_hit_cnt, history_cnt, all_hit_memo
        meas_axes[m].append(W)

        if history_cnt < m:
            all_hit_cnt += np.all(hit(pauli_ids, meas_axes[m - 1]), axis=1)
            history_cnt += 1

        term_1 = np.exp(-(epsilon**2) / 2 * all_hit_cnt)
        all_hit = all_hit_memo & hit(pauli_ids[:, k], meas_axes[m][k])
        term_2 = 1 - nu * all_hit * 3.0 ** (-weight(pauli_ids[:, k + 1 : n_qubit]))
        term_3 = (1 - nu * 3.0 ** (-weight(pauli_ids))) ** (meas_cnt - 1 - m)

        meas_axes[m].pop()

        return sum(term_1 * term_2 * term_3)

    for m in range(meas_cnt):
        all_hit_memo = np.ones(num_terms, dtype=bool)
        for k in range(n_qubit):
            conf_list = [calc_expected_conf(W, m, k) for W in range(1, 4)]
            meas_axes[m].append(np.argmin(conf_list) + 1)
            all_hit_memo &= hit(pauli_ids[:, k], meas_axes[m][k])

    return np.array(meas_axes)
