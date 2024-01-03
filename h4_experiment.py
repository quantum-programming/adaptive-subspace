import os
import pickle
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import datetime

from openfermion import get_fermion_operator, count_qubits, FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from pyscf import gto, scf

from molecule_info import MoleculeInfo
from subspace_expansion import SubspaceExpansion
from general_active_space import get_molecular_hamiltonian_generalAS
from sampler import local_dists_optimal


def _convert_spin_sector(ham_f):
    "convert |↑↓↑↓...> representation to |↑↑...↓↓...> representation"
    n = count_qubits(ham_f)
    n_half = int(n/2)
    index_map = {i: int(i/2) if i % 2 == 0 else n_half + int((i-1)/2) for i in range(n)}
    ham_convert = FermionOperator()
    ham_convert.terms = {tuple((index_map[i], j) for i, j in term): coef for term, coef in ham_f.terms.items()}
    return ham_convert


def _get_h4_hamiltonian(length):
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
    ham_f = _convert_spin_sector(ham_f)
    ham = jordan_wigner(ham_f)

    return ham


def simulate_energy_vs_interatomic_distance():
    energies_rigorous = []
    energies_2n1p_gs = []
    energies_2n1p_cisd = []
    params = {"molecule": "H4", "n_qubits": 8, "verbose": 0, "shots": 1e10, "n_lev": 20, "subspace": "2n1p", "spin_supspace": "up"}
    for length in tqdm(np.arange(0.5, 4.5, 0.1)):
        ham = _get_h4_hamiltonian(length)
        h4 = MoleculeInfo(params, ham)
        subspace_expansion = SubspaceExpansion(params, h4)
        energy_2n1p_gs = subspace_expansion.solve_qde_classically(h4, h4.state_gs)[1]
        energy_2n1p_cisd = subspace_expansion.solve_qde_classically(h4, h4.state_cisd)[1]
        
        energies_rigorous.append(h4.energy_excited)
        energies_2n1p_gs.append(energy_2n1p_gs)
        energies_2n1p_cisd.append(energy_2n1p_cisd)

    return energies_rigorous, energies_2n1p_gs, energies_2n1p_cisd


def _estimate_qse_matrix_elements_noise_simulator(beta_eff, params):
    def estimate_qse_matrix_element_noise_simulator(i, j, op_mat, result_mat):
        if j >= i:
            for term, coef in op_mat[i, j].terms.items():
                if coef == 0:
                    continue
                if term not in pauli_exp_dict:
                    pauli_exp_dict[term] = state.conj()@pauli_mat_dict[term]@state
                n_shot = get_n_shots_per_pauli(term, beta_eff, shots_total)
                if n_shot == 0:
                    continue
                shot_noise = np.random.normal() * coef * np.sqrt(abs(1-pauli_exp_dict[term]**2)/n_shot)
                result_mat[i, j] += coef*pauli_exp_dict[term] + shot_noise
        else:
            result_mat[i, j] = result_mat[j, i]
        return

    def get_n_shots_per_pauli(term, beta, shots_total):
        rate = np.prod([beta[i, {"X": 0, "Y": 1, "Z": 2}[label]] for i, label in term])
        return int(shots_total*rate)

    h_ij = params["h_ij"]
    s_ij = params["s_ij"]
    pauli_mat_dict = params["pauli_mat_dict"]
    state = params["molecule"].state_gs
    shots_total = params["subspace_expansion"].params["shots"]
    qse_dimension = max([i for i, _ in h_ij.keys()]) + 1
    h_eff = np.zeros([qse_dimension]*2, dtype=complex)
    s_mtrc = np.zeros([qse_dimension]*2, dtype=complex)

    pauli_exp_dict = {}
    for i, j in h_ij.keys():
        estimate_qse_matrix_element_noise_simulator(i, j, h_ij, h_eff)
        estimate_qse_matrix_element_noise_simulator(i, j, s_ij, s_mtrc)
    return h_eff, s_mtrc


def _iterative_run_lbcs(params):
    subspace_expansion = params["subspace_expansion"]
    hamiltonian = params["molecule"].hamiltonian
    n_qubit = subspace_expansion.n_qubit
    np.random.seed(params["seed"])

    dict_iter_lbcs = {
        "results_mat": [],
        "energy_excited": [],
        "alpha": [],
        "ops": [],
        "beta": [],
    }

    # get initial beta_eff
    h_d_jw, _ = subspace_expansion._get_h_s_d(
        subspace_expansion.alpha_cisd, hamiltonian
    )
    beta_eff = local_dists_optimal(
        h_d_jw, n_qubit, "diagonal", "lagrange"
        ).real
 
    for j in range(params["n_iteration"]):
        # update matrix element of general eigenvalue problem
        h_eff, s_mtrc = _estimate_qse_matrix_elements_noise_simulator(
            beta_eff, params
        )
        dict_iter_lbcs["results_mat"].append({"h_eff": h_eff, "s_mtrc": s_mtrc})

        # update alpha
        if subspace_expansion.params["n_lev"] == "auto":
            energy_excited, alpha = subspace_expansion._solve_regularized_gen_eig_with_best_n_lev(
                h_eff,
                s_mtrc,
                params["molecule"].energy_excited,
                threshold=1/np.sqrt(subspace_expansion.params["shots"]),
                return_vec=True,
            )
        else:
            energy_excited, alpha = subspace_expansion._solve_nlev_regularized_gen_eig(
                h_eff,
                s_mtrc,
                n_lev=subspace_expansion.params["n_lev"],
                threshold=1/np.sqrt(subspace_expansion.params["shots"]),
                return_vec=True,
            )
        dict_iter_lbcs["energy_excited"].append(energy_excited)
        dict_iter_lbcs["alpha"].append(alpha)
        if params["verbose"] >= 1:
            print(f"RUN[{params['seed']}-{j}]", "E excited (QSE) ", energy_excited)

        # update operator of H_d, S_d 
        h_d_jw_iter, s_d_jw_iter = subspace_expansion._get_h_s_d(
            alpha, params["molecule"].hamiltonian
        )
        dict_iter_lbcs["ops"].append({"H_d_jw": h_d_jw_iter, "S_d_jw": s_d_jw_iter})

        # update beta_eff
        beta_eff = local_dists_optimal(h_d_jw_iter, n_qubit, "diagonal", "lagrange").real
        dict_iter_lbcs["beta"].append({"beta_eff": beta_eff})
    return dict_iter_lbcs


def _get_pauli_mat_dict(file_name, n_qubits, h_ij, load=True):
    if load and file_name in os.listdir("h4_experiment_results"):
        with open(f"h4_experiment_results/{file_name}", "rb") as f:
            pauli_mat_dict = pickle.load(f)
    else:
        pauli_mat_dict = {
            term: get_sparse_operator(QubitOperator(term), n_qubits=n_qubits)
            for term in tqdm({term for op_sum in h_ij.values() for term in op_sum.terms.keys()})
        }
        with open(f"h4_experiment_results/{file_name}", "wb") as f:
            pickle.dump(pauli_mat_dict, f)
    return pauli_mat_dict


def simulate_qse_convergence():
    params_qse = {"molecule": "H4", "n_qubits": 8, "verbose": 0, "shots": 1e8, "n_lev": 20, "subspace": "2n1p", "spin_supspace": "up"}
    ham = _get_h4_hamiltonian(length=2.0)
    h4 = MoleculeInfo(params_qse, ham)
    subspace_expansion = SubspaceExpansion(params_qse, h4)
    h_ij, s_ij = subspace_expansion._get_h_s_ij(h4.hamiltonian)
    params_iteration = {
        "subspace_expansion": subspace_expansion,
        "molecule": h4,
        "h_ij": h_ij,
        "s_ij": s_ij,
        "pauli_mat_dict": _get_pauli_mat_dict("H4_8_length_2.0_pauli_mat_dict.pkl", params_qse["n_qubits"], h_ij, load=True),
        "n_iteration": 10,
        "verbose": 1,
        "cpu_assigned": 10,
    }
    params_iteration_list = [{"seed": i, **params_iteration} for i in range(params_iteration["n_iteration"])]
    pool = Pool(processes=params_iteration["cpu_assigned"])
    iterative_run_results = [result_dict for result_dict in pool.imap_unordered(_iterative_run_lbcs, params_iteration_list)]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")

    with open(f"h4_experiment_results/H4_8_2n1p_l2.0_{params_qse['shots']:.0e}_{timestamp}.pkl", "wb") as f:
        pickle.dump(iterative_run_results, f)

    energy_excited_exact = h4.energy_excited
    energy_excited_cisd = subspace_expansion.solve_qde_classically(h4, h4.state_cisd)[1]
    energies_excited = [result["energy_excited"] for result in iterative_run_results]
    return energy_excited_exact, energy_excited_cisd, energies_excited
