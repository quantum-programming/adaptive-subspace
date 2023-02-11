import math
from typing import Tuple

import numpy as np
from openfermion import count_qubits
from openfermion.ops import QubitOperator
from scipy.optimize import minimize
from scipy.optimize.optimize import OptimizeResult


class OverlappedGrouping(object):
    def __init__(self, op_hamiltonian: QubitOperator, T: int):
        """Initialize overlapped grouping class

        Args:
            op_hamiltonian (QubitOperator): observable or Hamiltonian
            T (int): the number of copies for the probability distribution generation
        """
        self.op_hamiltonian = op_hamiltonian
        self.T = T

    def get_meas_and_p(self) -> QubitOperator:
        """get measurement set and its probability distribution

        Returns:
            QubitOperator: efficient measuremnt set and its probability as openfermion operators.
            Note that probability distribution is interpreted as coefficients of operators.
        """
        MAX_STEP = 10
        observ = self._get_hamiltonian_from_openfermion(self.op_hamiltonian)
        [m, Nq] = observ.shape
        Nq = Nq - 1

        print(f"CutOGM, the number of qubit: {Nq}, the number of observables: {m}\n")

        # sort observables by its weights.
        observ = observ[abs(observ[:, 0]).argsort()][::-1]

        cur = 0  # current set number
        added = np.zeros(m)

        # get measurement group
        meas = {}
        pr = {}
        while True:
            j = 1
            while (j <= m) and (added[j - 1] > 0):
                j += 1
            if j > m:
                break
            cur += 1
            meas[cur] = observ[j - 1, 1:]
            added[j - 1] = 1
            sum_pr = abs(observ[j - 1, 0])
            start = j
            j += 1  # current observable

            #     initialize the set elements
            while j <= m:
                if self._if_commute(meas[cur], observ[j - 1, 1:]):
                    meas[cur] = self._update_meas(meas[cur], observ[j - 1, 1:], Nq)
                    added[j - 1] = 1
                    sum_pr += abs(observ[j - 1, 0])
                j += 1

            #     initize pr
            pr[cur] = sum_pr
            #     maximize the set
            for k in range(start):
                if self._if_commute(meas[cur], observ[k, 1:]):
                    meas[cur] = self._update_meas(meas[cur], observ[k, 1:], Nq)

        # optimize measurement probability
        len_ = cur
        pr_arr = np.array(list(pr.values()))
        sum_ = sum(pr_arr[:len_])
        pr_arr = pr_arr / sum_
        meas_arr = np.array(list(meas.values())).T
        exitflag = 1
        diag_new = 0

        for step in range(MAX_STEP):
            meas_and_p = np.vstack([meas_arr, pr_arr])
            meas_and_p = meas_and_p[:, meas_and_p[-1].argsort()][:, ::-1]
            meas_arr = meas_and_p[:-1, :]
            pr_arr = meas_and_p[-1, :]

            [new_len, new_pr] = self._cut_more_set(pr_arr)
            if self._variance(new_pr, observ, meas_arr) < self._variance(
                pr_arr, observ, meas_arr
            ):
                pr_arr = new_pr
                len_ = new_len
                meas_arr = meas_arr[:, :len_]

            elif exitflag == 0:
                break

            result = self._opt_diag_var(pr_arr, 10 * step, observ, meas_arr)
            [pr_arr, diagonal_var, exitflag] = [result.x, result.fun, result.status]

            if abs(diag_new - diagonal_var) < 1e-3:
                break

            diag_new = diagonal_var

        if not result.success:
            result = self._opt_diag_var(pr_arr, 100, observ, meas_arr)
            [pr_arr, diagonal_var, exitflag] = [result.x, result.fun, result.status]
            if not result.success:
                print(result)

        #   return meas_and_p
        return self._get_qubitoperator_from_meas_and_p(meas_and_p)

    def _cut_more_set(self, pr: dict) -> Tuple[int, dict]:
        """create overlapping group set

        Args:
            pr (dict): probability distribution of overlapping group set

        Returns:
            Tuple[int, dict]: number of set, probability distribution of measurement set
        """
        count = 0
        s = 0
        len_ = len(pr)
        pr_ = pr.copy()

        while (count < self.T) and (s < len_):

            sum_ = math.ceil(self.T * pr[s])
            s += 1
            for k in range(sum_):
                count += 1
                if count >= self.T:
                    break

        if s < len_:
            len_ = s

            pr_ = pr[:len_].copy()
            sum_ = 0
            for j in range(len_):
                sum_ = sum_ + pr_[j]
            for j in range(len_):
                pr_[j] = pr_[j] / sum_
        return len_, pr_

    def _update_meas(self, meas: np.ndarray, ele: np.ndarray, Nq: int) -> np.ndarray:
        """update i-th measurement set to incorporate j-th observable if compatible
            e.g:
            self._update_meas(meas=[0,3,3,3], ele=[3, 0, 0, 1], Nq = 4)
            >> [3, 3, 3, 3]

        Args:
            meas (np.ndarray): i-th measurement set
            ele (np.ndarray): j-th observable element
            Nq (int): the number of qubit to express Hamiltonian

        Returns:
            np.ndarray: updated i-th measurment set
        """
        new_meas = np.zeros(Nq)

        for j in range(Nq):
            if (meas[j] == 0) and ele[j] > 0:
                new_meas[j] = ele[j]
            else:
                new_meas[j] = meas[j]

        return new_meas

    def _variance(self, pr: np.ndarray, observ: np.ndarray, meas: np.ndarray) -> float:
        """calculate variance of estimator as loss function

        Args:
            pr (np.ndarray): probability distribution of overlapping group set
            observ (np.ndarray): observables with coefficient in matrix form
            meas (np.ndarray):  measurement set in matrix form

        Returns:
            float: the value of variance
        """
        len_pr = len(pr)

        [m, Nq] = observ.shape

        # diagonal variance %% if there exists an observable Q, which has no rela
        var = 0
        for j in range(m):  # observ[j]
            temp = 0
            for k in range(len_pr):  # meas[k]
                if self._if_commute(observ[j, 1: Nq + 1], meas[:, k]):
                    temp = temp + pr[k]
            if temp != 0:
                var = var + observ[j, 0] ** 2 / temp
            else:
                var = var + observ[j, 0] ** 2 * self.T
        return var

    def _if_commute(self, op1_int: np.ndarray, op2_int: np.ndarray) -> bool:
        """evaluate qubit-wise commutation of two operators

        Args:
            op1_int (np.ndarray): operator index
            op2_int (np.ndarray): operator index

        Returns:
            bool: whether two operators are commutable
        """
        res = True
        for i in range(len(op1_int)):
            if (op1_int[i] != op2_int[i]) and (op1_int[i] != 0) and (op2_int[i] != 0):
                res = False
                break
        return res

    def _opt_diag_var(
        self, pr: np.ndarray, iter_num: int, observ: np.ndarray, meas: np.ndarray
    ) -> OptimizeResult:
        """optimize probability distribution that minimize variance

        Args:
            pr (np.ndarray): probability distribution of overlapping group set
            iter_num (int): the number of iteration to run optimization
            observ (np.ndarray): observables with coefficient in matrix form
            meas (np.ndarray):  measurement set in matrix form

        Returns:
            OptimizeResult: optimized result with optimized probability, variance and status.
        """
        cons = {"type": "eq", "fun": lambda x: sum(x) - 1}
        options = {
            "maxiter": iter_num,
            "ftol": 0.0001,
        }
        result = minimize(
            fun=lambda x: self._variance(x, observ, meas),
            x0=pr,
            options=options,
            bounds=((0, 1),) * len(pr),
            constraints=cons,
            method="SLSQP",
        )

        return result

    def _get_hamiltonian_from_openfermion(self, operator: QubitOperator, num_qubit: int = None) -> np.ndarray:
        """convert operators in openfermion form into obserbables in matrix form

        Args:
            operator (QubitOperator): operators in openfermion form

        Returns:
            np.ndarray: obserbables in matrix form
        """
        pauli_dict = {"X": 1, "Y": 2, "Z": 3}
        op_terms = {k: v for k, v in operator.terms.items() if k != ()}
        num_ops = len(op_terms)
        if num_qubit is None:
            num_qubit = count_qubits(operator)
        observ = np.zeros((num_ops, num_qubit + 1))
        for idx, (ops, coef) in enumerate(op_terms.items()):
            observ[idx, 0] = coef
            for q_idx, pauli in ops:
                observ[idx, q_idx + 1] = pauli_dict[pauli]
        return observ

    def _get_qubitoperator_from_meas_and_p(self, meas_and_p: np.ndarray) -> QubitOperator:
        """convert  measurement and probability in matrix into openfermion operators.
        Note that probability distribution is interpreted as coefficients of operators.

        Args:
            meas_and_p (np.ndarray): measuremnt set and its probability

        Returns:
            QubitOperator: operators in openfermion form
        """
        pauli_dict = {1: "X", 2: "Y", 3: "Z"}
        ham = QubitOperator()
        for terms_and_coef in meas_and_p.T:
            terms = terms_and_coef[:-1]
            coef = terms_and_coef[-1]
            pauli_str = " ".join(
                [f"{pauli_dict[op]}{i}" for i, op in enumerate(terms) if op != 0]
            )
            ham += coef * QubitOperator(pauli_str)

        return ham
