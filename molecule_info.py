from typing import Dict
import os

import numpy as np
from scipy.sparse.linalg import eigsh
from openfermion import utils, count_qubits
from openfermion.linalg import get_ground_state, get_sparse_operator


class MoleculeInfo(object):
    def __init__(
      self,
      params: Dict,
      hamiltonian=None
    ):
        self.hamiltonian = hamiltonian if hamiltonian else self._load_hamiltonian(params)
        self.n_qubit = count_qubits(self.hamiltonian)
        self.sparse_hamiltonian = get_sparse_operator(self.hamiltonian)
        self.energy_gs, self.state_gs = get_ground_state(self.sparse_hamiltonian)
        self.hf_state_label = f"{np.argmax(abs(self.state_gs)):0{self.n_qubit}b}"
        self.state_cisd = self._prepare_cisd(self.sparse_hamiltonian, self.hf_state_label, params["verbose"])
        self.vec_args = self._particle_number_sector_spin(N_tot=self.n_qubit, M=self.hf_state_label.count("1")-1)
        self.ham_proj = self._get_partial_matrix(self.sparse_hamiltonian, self.vec_args)
        self.energy_excited = get_ground_state(self.ham_proj)[0]

    def _load_hamiltonian(self, params):

        assert "molecule" in params
        assert "n_qubits" in params

        molecule = params["molecule"]
        n_qubits = params["n_qubits"]
        file_name = f"{molecule}_{n_qubits}_jw.data"

        assert file_name in os.listdir("hamiltonians")

        hamiltonian = utils.load_operator(
            file_name=file_name,
            data_directory="hamiltonians",
            plain_text=True,
        )

        return hamiltonian

    def _prepare_cisd(self, ham_mat, hf_bitstring, print_mode):
        def hamming_dist(str1, str2):
            return sum([b1 != b2 for b1, b2 in zip(str1, str2)])

        def get_cisd_indices(hf_bitstring):
            n_particle = hf_bitstring.count("1")
            good_inds = []
            for i in range(2**self.n_qubit):
                bitstring = bin(i)[2:].zfill(self.n_qubit)
                if bitstring.count("1") == n_particle and hamming_dist(hf_bitstring, bitstring) <= 4:
                    good_inds.append(i)
            return good_inds

        cisd_inds = get_cisd_indices(hf_bitstring)
        ham_cisd = ham_mat[cisd_inds, :][:, cisd_inds]
        lambda_cisd, alpha_cisd = eigsh(ham_cisd, which="SA", k=self.n_qubit)
        if print_mode >= 1:
            print("energy=", lambda_cisd[0])
            print(f"HF state |{hf_bitstring}>")
        state = np.zeros(2**self.n_qubit, dtype=complex)
        for cisd_ind, value in zip(cisd_inds, alpha_cisd[:, 0]):
            state[cisd_ind] = value
        return state

    def _particle_number_sector_spin(self, N_tot, M):
        def count_particle_number(i, N_tot):
            return sum([int(b) for b in bin(i)[2:].zfill(N_tot)])

        args = [i for i in range(2**N_tot) if count_particle_number(i, N_tot) == M]
        return args

    def _get_partial_matrix(self, mat, args):
        return mat[:, args][args, :]
 