from typing import Dict
from multiprocessing import Pool
import os
import pickle
import json
import itertools

import scipy
import numpy as np
from scipy.linalg import eigh
from openfermion import utils, FermionOperator, QubitOperator
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import jordan_wigner

from molecule_info import MoleculeInfo
from overlapped_grouping import OverlappedGrouping
from qubit_wise_commuting import group_commuting
from derandomized import derandomized_classical_shadow
from sampler import (
    LocalPauliShadowSampler_core,
    get_samples,
    estimate_exp,
    estimate_exp_lbcs,
    estimate_exp_ogm,
    estimate_exp_derand,
    local_dists_optimal,
)


class SubspaceExpansion(object):
    def __init__(
        self,
        params: Dict,
        molecule: MoleculeInfo
    ):
        self.params = params
        self.n_qubit = molecule.n_qubit
        self.qse_ops = self._get_qse_ops()
        self.alpha_cisd = self.solve_qde_classically(molecule, molecule.state_cisd)[0]
        self.sampler = LocalPauliShadowSampler_core(
            self.n_qubit, 
            molecule.state_gs,
            params["shots"],
            nshot_per_axis=1
        )
        self.h_ij = None
        self.s_ij = None
        self.beta_eff = None
        self.ogm_meas_set = None

    def _get_qse_ops(self):
        # １粒子励起の演算子
        assert self.params["spin_supspace"] in ["up", "down", "all"]
        if self.params["spin_supspace"] == "up":
            iterator = range(int(self.n_qubit/2))
        elif self.params["spin_supspace"] == "down":
            iterator = range(int(self.n_qubit/2), self.n_qubit)
        elif self.params["spin_supspace"] == "all":
            iterator = range(self.n_qubit)
          
        assert self.params["subspace"] in ["1n", "2n1p"]
        if self.params["subspace"] == "1n":
            terms = [FermionOperator(f"{i}") for i in iterator]
        elif self.params["subspace"] == "2n1p":
            terms = [
                FermionOperator(f"{i} {j}^ {k}")
                for i, j, k in itertools.product(iterator, repeat=3)
            ]
        return [jordan_wigner(op) for op in terms]

    def solve_qde_classically(self, molecule, state):
        # preprocess (classical) subspace-expansion
        terms_eff_dict = {}
        for i, term in enumerate(self.qse_ops):
            terms_eff_dict[str(term)] = get_sparse_operator(term, n_qubits=self.n_qubit) @ state

        h_eff = np.zeros([len(terms_eff_dict)]*2)
        s_mtrc = np.zeros([len(terms_eff_dict)]*2)

        for i, P_i_psi_ful in enumerate(terms_eff_dict.values()):
            P_i_psi = P_i_psi_ful[molecule.vec_args]
            for j, P_j_psi_ful in enumerate(terms_eff_dict.values()):
                P_j_psi = P_j_psi_ful[molecule.vec_args]
                if i <= j:
                    h_eff[i, j] = (P_i_psi.conj() @ molecule.ham_proj @ P_j_psi).real
                    s_mtrc[i, j] = (P_i_psi.conj() @ P_j_psi).real
                else:
                    h_eff[i, j] = h_eff[j, i].conj()
                    s_mtrc[i, j] = s_mtrc[j, i].conj()

        # solve_nlev_regularized_gen_eigを使った方がいい場合を確認(多分2p1nのみ)
        try:
            _, alpha_all = eigh(h_eff, s_mtrc, eigvals_only=False)
            alpha = alpha_all[:, 0]
        except scipy.linalg.LinAlgError:
            try:
                _, alpha = self._solve_nlev_regularized_gen_eig(
                    h_eff,
                    s_mtrc,
                    n_lev=self.params["n_lev"],
                    threshold=1 / np.sqrt(self.params["shots"]),
                    return_vec=True,
                )
            except scipy.linalg.LinAlgError:
                raise ValueError(
                    "n_lev is too high given shots. Retry with lower n_lev"
                )
        
        energy_excited = (alpha.conj() @ h_eff @ alpha) / (alpha.conj() @ s_mtrc @ alpha)
        if self.params["verbose"] >= 1:
            print("E excited  (linalg)      ", molecule.energy_excited)
            print("E excited (QSE w/o noise)", energy_excited)

        return alpha, energy_excited

    # 修正予定
    def _get_h_s_d(self, alpha_se, ham):
        # complex_to_realの役割を確認中(浮動小数点のごみを消してる？)
        # def complex_to_real(ham):
        #     coefs = np.array(list(ham.terms.values()))
        #     ret = QubitOperator()
        #     if abs(coefs.imag).max() < 1e-6:
        #         ret.terms = {k: v for k, v in zip(ham.terms.keys(), coefs.real) if abs(v) > 1e-16}
        #     else:
        #         ret.terms = {k: v for k, v in ham.terms.items() if abs(v) > 1e-16}
        #     return ret

        def qubit_op_from_term(terms):
            op = QubitOperator()
            op.terms = terms
            return op
            
        # 変数の命名法を再検討
        P_mat = sum([a_j*P_j for a_j, P_j in zip(alpha_se, self.qse_ops)])
        # H_d_jw = complex_to_real(utils.hermitian_conjugated(P_mat) * ham * P_mat)
        # S_d_jw = complex_to_real(utils.hermitian_conjugated(P_mat) * P_mat)
        H_d_jw = utils.hermitian_conjugated(P_mat) * ham * P_mat
        S_d_jw = utils.hermitian_conjugated(P_mat) * P_mat
        H_d_jw = qubit_op_from_term({k: v for k, v in H_d_jw.terms.items() if abs(v) > 1e-15})
        S_d_jw = qubit_op_from_term({k: v for k, v in S_d_jw.terms.items() if abs(v) > 1e-15})

        return H_d_jw, S_d_jw
    
    def _get_h_s_ij(self, ham):
        h_ij = {}
        s_ij = {}
        for i, term_i in enumerate(self.qse_ops):
            for j, term_j in enumerate(self.qse_ops):
                if j >= i:
                    h_ij[i, j] = utils.hermitian_conjugated(term_i) * ham * term_j
                    s_ij[i, j] = utils.hermitian_conjugated(term_i) * term_j
                else:
                    h_ij[i, j] = h_ij[j, i]
                    s_ij[j, i] = s_ij[j, i]
        return h_ij, s_ij

    def estimate_qse_matrix_elements(self, meas_axes):
        def estimate_qse_matrix_element(i, j, op_mat, result_mat): # pauli_exp_dictは読み込めると仮定
            if j >= i:
                if self.params["method"] == "naive_LBCS":
                    result_mat[i, j] += estimate_exp_lbcs(
                        op_mat[i, j],
                        self.sampler,
                        self.beta_eff[i, j]
                    )
                    return
                for term, coef in op_mat[i, j].terms.items():
                    if coef == 0:
                        continue
                    if term not in pauli_exp_dict:
                        if self.params["method"] == "CS":
                            pauli_exp_dict[term] = estimate_exp(
                                QubitOperator(term),
                                self.sampler,
                                meas_axes,
                                samples
                            )
                        elif self.params["method"] == "LBCS":
                            pauli_exp_dict[term] = estimate_exp_lbcs(
                                QubitOperator(term),
                                self.sampler,
                                self.beta_eff,
                                meas_axes,
                                samples
                            )
                        elif self.params["method"] == "DCS":
                            pauli_exp_dict[term] = estimate_exp_derand(
                                QubitOperator(term),
                                self.sampler,
                                meas_axes,
                                samples
                            )
                        elif self.params["method"] in ["OGM", "qubit_wise_commuting"] :
                            pauli_exp_dict[term] = estimate_exp_ogm(
                                QubitOperator(term),
                                self.sampler,
                                self.ogm_meas_set,
                                meas_axes,
                                samples
                            )
                    result_mat[i, j] += coef*pauli_exp_dict[term]
            else:
                result_mat[i, j] = result_mat[j, i]
            return
        qse_dimension = len(self.qse_ops)
        samples = get_samples(self.sampler, meas_axes)
        H_eff = np.zeros([qse_dimension]*2, dtype=complex)
        S_mtrc = np.zeros([qse_dimension]*2, dtype=complex)

        pauli_exp_dict = {}
        for i in range(qse_dimension):
            for j in range(qse_dimension):
                estimate_qse_matrix_element(i, j, self.h_ij, H_eff)
                estimate_qse_matrix_element(i, j, self.s_ij, S_mtrc)

        return H_eff, S_mtrc

    def _solve_nlev_regularized_gen_eig(self, h, s, n_lev=None, threshold=1e-15, vec_norm_thresh=np.infty, return_vec=False):
        s_vals, s_vecs = scipy.linalg.eigh(s,)
        s_vecs = s_vecs.T
        if n_lev is not None:
            good_vecs = np.array([vec for val, vec in zip(s_vals[::-1][:n_lev], s_vecs[::-1][:n_lev]) if val >= threshold])
        else:
            good_vecs = np.array([vec for val, vec in zip(s_vals[::-1], s_vecs[::-1]) if val >= 0])
        h_reg = good_vecs.conj() @ h @ good_vecs.T
        s_reg = good_vecs.conj() @ s @ good_vecs.T
        eigvec_norms = [np.linalg.norm(vec) for vec in scipy.linalg.eigh(h_reg, s_reg)[1].T]
        lowest_stable_index = [i for i, x in enumerate(eigvec_norms) if x < vec_norm_thresh][0]
        sol = scipy.linalg.eigh(h_reg, s_reg)
        if return_vec:
            return sol[0][lowest_stable_index],  sol[1][:, lowest_stable_index]@good_vecs
        return sol[0][lowest_stable_index]

    def _solve_regularized_gen_eig_with_best_n_lev(
            self, h_eff, s_mtrc, true_energy_excited, threshold, return_vec):
        energies_excited = []
        alpha_list = []
        for i in range(1, len(self.qse_ops)):
            try:
                ret = self._solve_nlev_regularized_gen_eig(
                    h_eff,
                    s_mtrc,
                    n_lev=i,
                    threshold=threshold,
                    return_vec=return_vec,
                )
                if return_vec:
                    energy_excited, alpha_se = ret
                    alpha_list.append(alpha_se)
                else:
                    energy_excited = ret

                energies_excited.append(energy_excited)
            except scipy.linalg.LinAlgError:
                break
        best_idx = np.argmin(abs(np.array(energies_excited) - true_energy_excited))
        if self.params["verbose"] >= 2:
            print("best n_lev:", best_idx + 1)
        if return_vec:
            return energies_excited[best_idx], alpha_list[best_idx]
        else:
            return energies_excited[best_idx]

    def execute_statistics(self, molecule):
        self.h_ij, self.s_ij = self._get_h_s_ij(molecule.hamiltonian)
        h_eff_list = []
        s_mtrc_list = []
        h_d_jw, _ = self._get_h_s_d(self.alpha_cisd, molecule.hamiltonian)
        if self.params["method"] == "CS":
            meas_axes_list = [
                self.sampler.generate_random_measurement_axis()
                for _ in range(self.params["n_trial"])
            ]
        elif self.params["method"] == "LBCS":
            beta_eff = local_dists_optimal(
                h_d_jw, self.n_qubit, "diagonal", "lagrange"
            ).real
            self.beta_eff = beta_eff
            meas_axes_list = [
                self.sampler.generate_random_measurement_axis(lbcs_beta=self.beta_eff)
                for _ in range(self.params["n_trial"])
            ]
        elif self.params["method"] == "DCS":
            meas_axes_base = derandomized_classical_shadow(h_d_jw, 1000, self.n_qubit)
            meas_axes = np.tile(
                meas_axes_base, 
                reps=(int(self.params["shots"]/len(meas_axes_base)), 1)
            )
            meas_axes_list = [meas_axes]*self.params["n_trial"]
        elif self.params["method"] == "OGM":
            file_name = f'{self.params["molecule"]}_{self.n_qubit}_T{self.params["OGM_param_T"]}.data'
            if self.params["load"] and file_name in os.listdir("ogm_meas_sets"):
                ogm_meas_set = utils.load_operator(
                    file_name=file_name,
                    data_directory="ogm_meas_sets",
                    plain_text=True
                )
            else:
                ogm_meas_set = OverlappedGrouping(
                    h_d_jw, T=self.params["OGM_param_T"]
                ).get_meas_and_p()
                if self.params["verbose"] >= 1:
                    print("OGM measurement optimization has done")
                utils.save_operator(
                    operator=ogm_meas_set,
                    file_name=file_name,
                    data_directory="ogm_meas_sets",
                    plain_text=True,
                    allow_overwrite=True,
                )

            self.ogm_meas_set = ogm_meas_set
            meas_axes_list = [
                self.sampler.generate_random_measurement_axis(ogm_meas_set=ogm_meas_set)
                for _ in range(self.params["n_trial"])
            ]
        elif self.params["method"] == "qubit_wise_commuting":
            grouped_meas_set = group_commuting(h_d_jw)
            self.ogm_meas_set = grouped_meas_set
            meas_axes_list = [
                self.sampler.generate_random_measurement_axis(ogm_meas_set=grouped_meas_set)
                for _ in range(self.params["n_trial"])
            ]
        elif self.params["method"] == "naive_LBCS":
            beta_eff = {}
            for (i, j), h_ij_op in self.h_ij.items():
                if j < i:
                    continue
                h_ij_op_real = QubitOperator()
                h_ij_op_real.terms = {k: abs(v) for k, v in h_ij_op.terms.items()}
                beta_eff[i, j] = local_dists_optimal(
                    h_ij_op_real, self.n_qubit, "diagonal", "lagrange"
                ).real
            self.beta_eff = beta_eff
            self.sampler.Ntot = int(self.params["shots"]/len(self.qse_ops)**2)
            # In naive_LBCS, meas_axes_list is not used but need to be defined
            meas_axes_list = [
                self.sampler.generate_random_measurement_axis()
            ]*self.params["n_trial"]

        pool = Pool(processes=self.params["cpu_assigned"])
        for row in pool.imap_unordered(self.estimate_qse_matrix_elements, meas_axes_list):
            h_eff, s_mtrc = row
            h_eff_list.append(h_eff)
            s_mtrc_list.append(s_mtrc)

        if self.params["write_result_matrix"]:
            dir_name = f'./result_matrix/{self.params["molecule"]}_'\
             + f'{self.n_qubit}_{self.params["shots"]}_'\
             + f'{self.params["method"]}_{self.params["suffix"]}'
            os.makedirs(f"{dir_name}", exist_ok=True)
            with open(f"{dir_name}/h_eff_list.pkl", "wb") as f:
                pickle.dump(h_eff_list, f)
            with open(f"{dir_name}/s_mtrc_list.pkl", "wb") as f:
                pickle.dump(s_mtrc_list, f)
            with open(f"{dir_name}/params.json", "w") as f:
                json.dump(self.params, f)
        if self.params["n_lev"] == "auto":
            E_list = [
                self._solve_regularized_gen_eig_with_best_n_lev(
                    h_eff,
                    s_mtrc,
                    molecule.energy_excited,
                    threshold=1/np.sqrt(self.params["shots"]),
                    return_vec=False,
                )
                for h_eff, s_mtrc in zip(h_eff_list, s_mtrc_list)
            ]
        else:
            try:
                E_list = [
                    self._solve_nlev_regularized_gen_eig(
                        h_eff,
                        s_mtrc,
                        n_lev=self.params["n_lev"],
                        threshold=1/np.sqrt(self.params["shots"]),
                        return_vec=False,
                    )
                    for h_eff, s_mtrc in zip(h_eff_list, s_mtrc_list)
                ]
            except scipy.linalg.LinAlgError:
                raise ValueError(
                    "n_lev is too high given shots. Retry with lower n_lev"
                )
        err = abs(np.array(E_list) - molecule.energy_excited).mean().round(4)
        std = np.std(E_list, ddof=1).round(4)
        if self.params["verbose"] >= 1:
            print("energies_excited: \n", np.array(E_list).round(4))
        return err, std
