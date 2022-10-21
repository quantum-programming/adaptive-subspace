import openfermion
import glob

# Load hamiltonian
molecule_type = "H2"
basis_type = "sto-6g"
conversion_type = "jordan-wigner"
directory = glob.glob(f"{molecule_type}_{basis_type}*")[0]

if conversion_type == "jordan-wigner":
    ham_q = openfermion.load_operator(file_name = "jw_hamiltonian.data", data_directory = directory)
elif conversion_type == "bravyi-kitaev":
    ham_q = openfermion.load_operator(file_name = "bk_hamiltonian.data", data_directory = directory)

print("loaded hamiltonian of ")

# Get ground state energy
from openfermion import get_sparse_operator, count_qubits
import numpy as np
import scipy

n_qubit = count_qubits(ham_q)
print(f"...loaded N={n_qubit}qubit Hamiltonian of {molecule_type}, {basis_type}")

# convert into sparse Hamiltonian
ham_sp = get_sparse_operator(ham_q)
vals, vecs = scipy.sparse.linalg.eigsh(ham_sp, which = "SA", k= 10)
gsvec = vecs[:, 0]    

# ground state energy
print(f"ground state energy = {vals[0]:.5f}")

# double check
print(f"< GS | Ham | GS > = {(gsvec.conj() @ ham_sp @ gsvec).real:.5f}")