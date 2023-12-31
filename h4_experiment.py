from tqdm import tqdm
import numpy as np
from openfermion import get_fermion_operator, count_qubits, FermionOperator
from openfermion.transforms import jordan_wigner
from pyscf import gto, scf

from molecule_info import MoleculeInfo
from subspace_expansion import SubspaceExpansion
from general_active_space import get_molecular_hamiltonian_generalAS


def convert_spin_sector(ham_f):
    "convert |↑↓↑↓...> representation to |↑↑...↓↓...> representation"
    n = count_qubits(ham_f)
    n_half = int(n/2)
    index_map = {i: int(i/2) if i % 2 == 0 else n_half + int((i-1)/2) for i in range(n)}
    ham_convert = FermionOperator()
    ham_convert.terms = {tuple((index_map[i], j) for i, j in term): coef for term, coef in ham_f.terms.items()}
    return ham_convert


def get_h4_hamiltonian(length):
    geom = [("H", (-length/2*3, 0, 0)), ("H", (-length/2, 0, 0)),("H", (length/2, 0, 0)), ("H", (length/2*3, 0, 0))]
    basis_type = "sto-6g"

    mol = gto.M(atom=geom, basis=basis_type)
    # SCF波動関数のオブジェクトを生成。ここではRHFを使用する。
    mf = scf.RHF(mol)
    mf.verbose = 0
    # SCF計算の実行, エネルギーが得られる(-74.96444758277)
    mf.run()

    molecular_ham = get_molecular_hamiltonian_generalAS(
        mol, pyscf_mf=mf
    )

    # molecular_ham = molecule.get_molecular_hamiltonian(occ_inds, act_inds)
    ham_f = get_fermion_operator(molecular_ham)
    ham_f = convert_spin_sector(ham_f)
    ham = jordan_wigner(ham_f)

    return ham


def simulate_energy_vs_interatomic_distance():
    energies_rigorous = []
    energies_2n1p_gs = []
    energies_2n1p_cisd = []
    params = {"molecule": "H4", "n_qubits": 8, "verbose": 0, "shots": 1e10, "n_lev": 20, "subspace": "2n1p", "spin_supspace": "up"}
    for length in tqdm(np.arange(0.5, 4.5, 0.1)):
        ham = get_h4_hamiltonian(length)
        h4 = MoleculeInfo(params, ham)
        subspace_expansion = SubspaceExpansion(params, h4)
        energy_2n1p_gs = subspace_expansion.solve_qde_classically(h4, h4.state_gs)[1]
        energy_2n1p_cisd = subspace_expansion.solve_qde_classically(h4, h4.state_cisd)[1]
        
        energies_rigorous.append(h4.energy_excited)
        energies_2n1p_gs.append(energy_2n1p_gs)
        energies_2n1p_cisd.append(energy_2n1p_cisd)
        
    return energies_rigorous, energies_2n1p_gs, energies_2n1p_cisd