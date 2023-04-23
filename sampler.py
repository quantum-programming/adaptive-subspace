from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
# from derandomized import derandomized_classical_shadow
from openfermion import QubitOperator
from qulacs import QuantumCircuit, QuantumState

from lbcs_opt.var_opt_lagrange import find_optimal_beta_lagrange
from lbcs_opt.var_opt_scipy import find_optimal_beta_scipy
from overlapped_grouping import OverlappedGrouping
from utils import *


class LocalPauliShadowSampler_core(object):
    def __init__(
        self,
        n_qubit: int,
        state: QuantumState,
        Nshadow_tot: int,
        nshot_per_axis=1,
    ):
        self.n_qubit = n_qubit
        self.state = state
        self.Ntot = Nshadow_tot
        self.m = nshot_per_axis

    def set_state(self, state):
        self.state = state

    def set_random_seed(self, seed):
        np.random.seed(seed)

    def generate_random_measurement_axis(
        self,
        lbcs_beta: Optional[Iterable[Iterable]] = None,
        ogm_meas_set: Optional[Iterable[Iterable]] = None,
    ):
        """
        Creates a list of random measurement axis.
        For 3-qubit system, this looks like, e.g.,
        [1, 3, 2],
        which tells you to measure in
        [X, Z, Y]
        basis.

        return:
            List[n_qubit]
        """
        assert not (
            (lbcs_beta is not None) and (ogm_meas_set is not None)
        ), "you must choose either of lbcs or ogm"
        if lbcs_beta is not None:
            # meas_axes = [
            #     np.random.multinomial(1, beta_i, size=1)[0].argmax() + 1
            #     for beta_i in lbcs_beta
            # ]
            meas_axes = np.vstack([np.random.multinomial(1, beta_i, size=self.Ntot).argmax(axis=1) + 1 for beta_i in lbcs_beta]).T
        elif ogm_meas_set is not None:
            # pandasの処理をnp処理に変更できるかも
            df = pd.DataFrame(
                [
                    create_pauli_id_from_openfermion(op, self.n_qubit)[::-1]
                    for op in ogm_meas_set.terms
                ]
            )
            meas_axes_idx = np.random.multinomial(
                1, list(ogm_meas_set.terms.values()), size=self.Ntot
            ).argmax(axis=1)
            meas_axes = df.iloc[meas_axes_idx].values
        else:
            # meas_axes = list(np.random.randint(1, 4, size=self.n_qubit))
            meas_axes = np.random.randint(1, 4, size=(self.Ntot, self.n_qubit))
        return meas_axes
        

    def _sample_digits(self, meas_axis, nshot_per_axis = 1):
        """
        returns the measurement result at meas_axis.
        attributes:
            meas_axis: List of axes
            nshot_per_axis: number of measurement per axis
        """
        meas_state = QuantumState(self.n_qubit)
        meas_state.load(self.state)

        meas_circuit = QuantumCircuit(self.n_qubit)

        # それぞれのqubitに測定用のuntiaryを作用
        for qindex in range(self.n_qubit):
            _axis = meas_axis[qindex]
            if _axis == 1:
                # X軸測定のためのユニタリ
                meas_circuit.add_H_gate(qindex)
            elif _axis == 2:
                # Y軸測定のためのユニタリ
                meas_circuit.add_Sdag_gate(qindex)
                meas_circuit.add_H_gate(qindex)

        meas_circuit.update_quantum_state(meas_state)

       # サンプリングに対応するシミュレーション
        digits = meas_state.sampling(nshot_per_axis)
        return digits


def local_dists_optimal(ham, num_qubits, objective, method, β_initial=None, bitstring_HF=None):
        """Find optimal probabilities beta_{i,P} and return as dictionary
        attn: qiskit ordering"""
        assert objective in ['diagonal', 'mixed']
        assert method in ['scipy', 'lagrange']

        dic_tf = {
            "".join([{0:"I",1:"X",2:"Y",3:"Z"}[s] for s in create_pauli_id_from_openfermion(k, num_qubits)]): v
            for k, v in ham.terms.items() if len(k)>0
        }
        if method == 'scipy':
            beta_opt =  find_optimal_beta_scipy(dic_tf, num_qubits, objective,
                                           β_initial=β_initial, bitstring_HF=bitstring_HF)
        else:
            # method == 'lagrange'
            beta_opt = find_optimal_beta_lagrange(dic_tf, num_qubits, objective,
                                              tol=1.0e-5, iter=10000,
                                              β_initial=β_initial, bitstring_HF=bitstring_HF)

        return np.array(list(reversed(list(beta_opt.values())))).round(4)


def get_samples(
    sampler: LocalPauliShadowSampler_core, meas_axes: Iterable[Iterable]
) -> np.ndarray:  
    """
    Create a measurement sample according to measurement axes

    Args:
        sampler (LocalPauliShadowSampler_core): Sampler Class
        meas_axes (Iterable[Iterable]): measurement axes shaped as (total shot, num_qubit)

    Returns:
        np.ndarray: sampling result shaped as (total shot, num_qubit)
    """

    sample_digits = [
        sampler._sample_digits(_meas_ax, nshot_per_axis=sampler.m)
        for _meas_ax in meas_axes
    ]
    sample_digits = sum(sample_digits, [])  # 整形
    bitstring_array = [
        format(_samp, "b").zfill(sampler.n_qubit) for _samp in sample_digits
    ]
    samples = np.array(
        [[int(_b) for _b in _bitstring][::-1] for _bitstring in bitstring_array]
    )
    return samples


def estimate_exp(operator: QubitOperator, sampler: LocalPauliShadowSampler_core) -> float:
    """
    Estimate expectation value of Observable for Basic Classical Shadow

    Args:
        operator (QubitOperator): Observable such as Hamiltonian
        sampler (LocalPauliShadowSampler_core): Sampler Class 

    Returns:
        float: Expectation value
    """
    meas_axes = sampler.generate_random_measurement_axis()
    samples = get_samples(sampler, meas_axes)
    assert np.array(meas_axes).shape == np.array(samples).shape
    
    exp = 0
    for op, coef in operator.terms.items():
        
        pauli_ids = create_pauli_id_from_openfermion(op, sampler.n_qubit)[::-1]
        pauli = np.tile(pauli_ids, (sampler.Ntot,1))
        
        # This is the core of estimator, which corresponds to Algotihm 1 of
        # https://arxiv.org/abs/2006.15788
        arr = (np.array(meas_axes) == np.array(pauli)) * 3
        arr = (-1) ** np.array(samples) * arr 
        arr += (np.array(pauli) == 0)
        val_array =  np.prod(arr, axis = -1)

        exp += coef * np.mean(val_array)        

    return exp

def estimate_exp_lbcs(
    operator: QubitOperator,
    sampler: LocalPauliShadowSampler_core,
    beta: Iterable[Iterable],
    meas_axes: Iterable[Iterable] = None,
    samples: np.ndarray = None,
) -> float:
    """
    Estimate expectation value of Observable for Locally Biased Classical Shadow

    Args:
        operator (QubitOperator): Observable such as Hamiltonian
        sampler (LocalPauliShadowSampler_core): Sampler Class 
        beta (Iterable[Iterable]): weighted bias

    Returns:
        float: Expectation value
    """

    if meas_axes is None:
        meas_axes = sampler.generate_random_measurement_axis(lbcs_beta=beta)
    if samples is None:
        samples = get_samples(sampler, meas_axes)
    assert np.array(meas_axes).shape == np.array(samples).shape

    exp = 0
    for op, coef in operator.terms.items():
        pauli_ids = create_pauli_id_from_openfermion(op, sampler.n_qubit)[::-1]
        pauli = np.tile(pauli_ids, (sampler.Ntot,1))

        ############################
        # estimate expectation value
        ############################
        # beta_p_iの計算をもう少し早くできそう 
        beta_p_i = beta[range(sampler.n_qubit), meas_axes-1]
        # beta_p_i = np.array([[beta[i][pauli-1] for i,pauli in enumerate(meas_axes_row)] for meas_axes_row in meas_axes])
        # arr = (np.array(meas_axes) == np.array(pauli)) * (beta_p_i+1e-6 )**-1 
        arr = (np.array(meas_axes) == np.array(pauli)) * np.reciprocal(beta_p_i, where=beta_p_i != 0)
        arr = (-1) ** np.array(samples) * arr 
        arr += (np.array(pauli) == 0) 
        val_array =  np.prod(arr, axis = -1)

        exp += coef* np.mean(val_array)

    return exp


def estimate_exp_ogm(
    operator: QubitOperator,
    sampler: LocalPauliShadowSampler_core,
    meas_dist: Iterable[Iterable], 
    meas_axes: Iterable[Iterable] = None,
    samples: np.ndarray = None,
) -> float :
    """
    Estimate expectation value of Observable for Locally Biased Classical Shadow

    Args:
        operator (QubitOperator): Observable such as Hamiltonian
        sampler (LocalPauliShadowSampler_core): Sampler Class 
        meas_dist (Iterable[Iterable]): measurement set and its distribution 
                                        input is format of QubitOperator.terms

    Returns:
        float: Expectation value
    """
    
    def get_chi(grouper, q_i, pr, meas):
            return sum([p for p, m in zip(pr, meas) if  grouper._if_commute(q_i, m)])

    if meas_axes is None:
        meas_axes = sampler.generate_random_measurement_axis(ogm_meas_set=meas_dist)
    if samples is None:
        samples = get_samples(sampler, meas_axes)
    assert np.array(meas_axes).shape == np.array(samples).shape
    
    # precomuted values
    grouper = OverlappedGrouping(None, None)
    meas_as_arr = grouper._get_hamiltonian_from_openfermion(meas_dist, num_qubit=sampler.n_qubit)
    pr = meas_as_arr[:, 0]
    meas = meas_as_arr[:, 1:][:, ::-1]
    
    pauli_ids_list = [
        create_pauli_id_from_openfermion(op, sampler.n_qubit)[::-1] for op in operator.terms
    ]
    chi_dict = {
        tuple(pauli_id): get_chi(grouper, pauli_id, pr, meas) for pauli_id in pauli_ids_list
    }
    delta_dict = {
        tuple(pauli_id): {tuple(m): grouper._if_commute(pauli_id, m) for m in meas}
        for pauli_id in pauli_ids_list
    }
    samples_pm = (-1) **(np.array(samples))
    
    exp = 0
    for pauli_ids, coef in zip(pauli_ids_list, operator.terms.values()):
        ############################
        # estimate expectation value
        ############################
        if chi_dict[tuple(pauli_ids)] == 0:
            val_array = 0
        else: 
            val_array =  chi_dict[tuple(pauli_ids)]**(-1) * np.array([delta_dict[tuple(pauli_ids)][tuple(axes)] for axes in meas_axes])

        val_array *= np.prod(samples_pm[:, np.array(pauli_ids) != 0], axis = -1)     
        exp += coef* np.mean(val_array)

    return exp

