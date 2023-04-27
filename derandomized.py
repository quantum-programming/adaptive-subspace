from ast import operator
import matplotlib.pyplot as plt
import numpy as np
import math 

from qulacs import QuantumState, QuantumCircuit
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text
from openfermion import QubitOperator

from typing import List
from utils import create_pauli_id_from_openfermion


def derandomized_classical_shadow(
    operators: List[QubitOperator], meas_cnt: int, n_qubit: int
):
    # 0:'I', 1:'X', 2:'Y', 3:'Z'
    pauli_ids = [create_pauli_id_from_openfermion(op, n_qubit)[::-1] for op in operators.terms] 
    num_terms = len(operators.terms)

    meas_axes = [[] for i in range(meas_cnt)]

    def hit(a, b):
        return a == 0 or a == b

    def weight(a: List[int]):
        return len(a) - a.count(0)

    all_hit_cnt = [0 for i in range(num_terms)]
    history_cnt = [0 for i in range(num_terms)]
   
    all_hit_memo = [1] * num_terms

    # m は単調増加なのでメモして高速化できる
    def calc_expected_conf(W: int, m: int, k: int):

        nonlocal all_hit_cnt, history_cnt, all_hit_memo
        meas_axes[m].append(W)
        epsilon = 0.95
        res = 0
        nu = 1 - math.exp(-(epsilon**2) / 2)
        for l in range(num_terms):
            if history_cnt[l] < m:
                tmp = 1
                for k2 in range(n_qubit):
                    tmp *= hit(pauli_ids[l][k2], meas_axes[m - 1][k2])
                all_hit_cnt[l] += tmp
                history_cnt[l] += 1
            term_1 = math.exp(-(epsilon**2) / 2 * all_hit_cnt[l])

            # ここも一応出来るが、やるか検討中　早くはなる
            # all_hit = 1
            # for k2 in range(k + 1):
            #     all_hit *= hit(pauli_ids[l][k2], meas_axes[m][k2])
            all_hit = all_hit_memo[l] & hit(pauli_ids[l][k], meas_axes[m][k])
            term_2 = 1 - nu * all_hit * 3 ** (-weight(pauli_ids[l][k + 1 : n_qubit]))
            term_3 = (1 - nu * 3 ** (-weight(pauli_ids[l]))) ** (meas_cnt - 1 - m)
            res += term_1 * term_2 * term_3

        meas_axes[m].pop()

        return res

    hit_operator = [False] * num_terms
    for m in range(meas_cnt):
        all_hit_memo = [1] * num_terms
        for k in range(n_qubit):
            conf_list = []
            for W in range(1, 4):
                conf_list.append(calc_expected_conf(W, m, k))
            for i in range(3):
                if conf_list[i] == min(conf_list):
                    meas_axes[m].append(i + 1)
                    for l in range(num_terms):
                        all_hit_memo[l] &= hit(pauli_ids[l][k], meas_axes[m][k])
                    break
        # success = True
        # for i in range(len(operators)):
        #     if not hit_operator[i]:
        #         all_hit = True
        #         for k in range(n_qubit):
        #             if not hit(pauli_ids[i][k], meas_axes[m][k]):
        #                 all_hit = False
        #                 break
        #         hit_operator[i] |= all_hit
        # if sum(hit_operator) == len(operators):
        #     print(m + 1)
        #     return meas_axes[: (m + 1)]
    return np.array(meas_axes)
