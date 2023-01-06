import itertools
import re

import numpy as np
import pyscf
import scipy
from openfermion import bravyi_kitaev, get_sparse_operator
from openfermion.ops import FermionOperator, QubitOperator
from pyscf import scf, symm
from qulacs import QuantumState

#from .qse import hermitianize
def hermitianize(mat):
    if np.abs(np.triu(mat) - np.tril(mat).conj().T).sum() > 1e-5:
        if np.abs(np.triu(mat)).sum() > np.abs(np.tril(mat)).sum():
            return np.triu(mat) + np.triu(mat).conj().T - np.diag(mat.diagonal())
        else:
            return np.tril(mat) + np.tril(mat).conj().T - np.diag(mat.diagonal())

    else:
        # do nothing if it is already hermitian
        return mat

##########################################
# useful tools
##########################################


def get_hf_state(raw_bitstring, encoding="jordan-wigner"):
    if encoding == "jordan-wigner":
        bitstring = raw_bitstring
    elif encoding == "bravyi-kitaev":

        op = FermionOperator("")
        n_qubit = len(raw_bitstring)
        for i in range(n_qubit):
            op += (-1) ** (raw_bitstring[i] == "0") * FermionOperator("%d^ %d" % (i, i))
            # op = op * FermionOperator("%d^ %d"%(i, i))

        op_bk = bravyi_kitaev(op)
        sp_bk = get_sparse_operator(op_bk)
        hf_argument_bk = np.argmax(sp_bk.diagonal())
        bitstring = bin(hf_argument_bk)[2:]


    n_qubit = len(bitstring)
    state = QuantumState(n_qubit)
    hf_idx = int(bitstring, base=2)
    state.set_computational_basis(hf_idx)

    return state.get_vector()


def generate_hf_bitstring(molecule, act_inds=None, act_spinorbitals=None):
    assert not (
        act_inds is not None and act_spinorbitals is not None
    ), "take either act_inds or act_spinorbitals"

    # Set electron configure

    if isinstance(molecule, pyscf.gto.Mole):
        n_orb = molecule.nao
        hf_filled_spinorbitals = sorted(
            [2 * i for i in range(molecule.nelec[0])]
            + [2 * i + 1 for i in range(molecule.nelec[1])]
        )
    else:
        n_orb = molecule.n_orbitals
        n_elec = molecule.n_electrons
        hf_filled_spinorbitals = range(n_elec)

    # Set active space info
    if act_inds is None and act_spinorbitals is None:
        act_spinorbitals = range(n_orb * 2)
    elif act_spinorbitals is not None:
        assert (
            sorted(act_spinorbitals) == act_spinorbitals
        ), "act_spinorbitals must be sorted. Check the ordering of `partition`"
    elif act_inds is not None:
        act_spinorbitals = sum([[2 * i, 2 * i + 1] for i in act_inds], [])

    bs = ""
    for i in range(n_orb * 2):
        if i in act_spinorbitals:
            if i in hf_filled_spinorbitals:
                bs += "1"
            else:
                bs += "0"
    return bs


########################################
# For Hubbard, Fermionic operators from sorting of "score"
########################################
def generate_fermionic_excitation_terms_Hubbard(
    score_tensor,
    partition=None,
    mask_intra=True,
    n_set=None,
    allow_zero=False,
    minimum_partition_involvement=None,
):
    if mask_intra:
        score_tensor = mask_intrasubgraph_elements(score_tensor, partition)
    dim = score_tensor.shape[0]
    n_qubit = score_tensor.shape[0]
    tensor_rank = len(score_tensor.shape)
    n_nonzero = np.sum(
        np.abs(score_tensor) >= 1e-8 * (not allow_zero),
    )
    arg_sorted = np.argsort(np.abs(score_tensor), axis=None)[::-1][:n_nonzero]
    # terms = ["%d^ %d"%(_arg//n_qubit, _arg%n_qubit) for _arg in arg_sorted]

    # argument -> index set
    # e.g. 8901  -> 8 9 0 1
    # e.g. 8903 -> 8 9 0 3
    # if hf_bitstring = 1100000000, n_qubit = 10
    indset = [_convert_arg_to_tensor_index(_arg, dim, tensor_rank) for _arg in arg_sorted]
    indset = [_ind for _ind in indset if len(set(_ind)) == tensor_rank]
    if minimum_partition_involvement is not None:
        indset = [
            _ind
            for _ind in indset
            if _involved_partition_number(_ind, partition) >= minimum_partition_involvement
        ]

    terms = []
    for _inds in indset:
        terms += fermionic_excitation_term_from_tensorind(_inds)

    if n_set is not None:
        return terms[:n_set]
    else:
        return terms


def _involved_partition_number(tensor_indices, partition):
    parts = [which_subsystem(i, partition) for i in tensor_indices]
    return np.unique(parts).shape[0]


##########################################
# Fermionic operators from sorting of "score"
##########################################


def generate_fermionic_excitation_terms(
    fermionic_hamiltonian,
    hf_bitstring,
    excitation_order: int,
    #return_unscreened_term = False,
    n_term=2000,
    allow_zero=False,
    sz_symmetry=True,
    refer_to_HF=True,
    molecule_type=None,
    mol=None,
    act_inds=None,
    groups=None,
    characters=None,
    excitation_upperbound=None,
    impose_point_group_symmetry = False,
    return_score_tensor = False 
):
    """
    generate QSE operators according to the absolute amplitude of t1/t2 in Hamiltonian.
    attributes:
        fermionic_hamiltonian: openfermion operator
        hf_bitstring: Hartree-Fock occupation
        excitation_order         : 1 or 2, (t1 or t2)
        n_term          : number of excitation operators
        allow_zero : include (i, j) with zero amplitude or not
        sz_symmetry : impose spin conservation
        refer_to_HF :  exclude c_i^ c_j with j \in unoccupied sites
        molecule_type : string
        impose_point_group_symmetry: bool
    """

    ### Generate score tensor
    if excitation_order == 1:
        t1 = extract_hopping_matrix(fermionic_hamiltonian)
        score_tensor = np.abs(np.copy(t1))
    elif excitation_order == 2:
        # t2 =
        score_tensor = np.abs(
            extract_interaction_tensor(
                fermionic_hamiltonian,
                two_elec_hopping_only=True,
                refer_to_HF=refer_to_HF,
                hf_bitstring=hf_bitstring,
            )
        )
    elif excitation_order == 3:
        # raise Exception("excitation_order must be 1 or 2.")
        # assert excitation_upperbound is not None, "Take some value for excitation_upperbound"
        return _generate_fermionic_excitation_terms_three_body(
            fermionic_hamiltonian,
            hf_bitstring,
            n_term,
            refer_to_HF,
            excitation_upperbound=excitation_upperbound,
        )

    else:
        raise Exception("excitation_order must be 1 or 2.")

    assert not (
        score_tensor < 0
    ).any(), "input must be non-negative for the current implementation."
    #if mask_intra:
        #score_tensor = mask_intrasubgraph_elements(score_tensor, partition)

    terms = sort_terms_by_score(
        score_tensor,
        hf_bitstring = hf_bitstring,
        n_term = n_term,
        allow_zero=allow_zero,
        sz_symmetry=sz_symmetry,
        refer_to_HF=refer_to_HF,
        molecule_type=molecule_type,
        mol=mol,
        act_inds=act_inds,
        groups=groups,
        characters=characters,
        impose_point_group_symmetry = impose_point_group_symmetry
    )

    if return_score_tensor:
        return terms, score_tensor
    return terms

def sort_terms_by_score(
    score_tensor,
    n_term = 20,
    allow_zero = False,
    #return_unscreened_term = False,
    hf_bitstring = None,
    refer_to_HF  = False,
    sz_symmetry = None,
    molecule_type=None,
    mol=None,
    act_inds=None,
    groups=None,
    characters=None,
    impose_point_group_symmetry = False,
    verbose = 0
):

    dim = score_tensor.shape[0]
    tensor_rank = len(score_tensor.shape)
    n_nonzero = np.sum(
        np.abs(score_tensor) >= 1e-8 * (not allow_zero),
    )
    arg_sorted = np.argsort(np.abs(score_tensor), axis=None)[::-1][:n_nonzero]

    inds_set = []
    sorted_inds_set = []
    terms = []
    _n = 0

    while _n < len(arg_sorted):
        _arg = arg_sorted[_n]

        # argument -> index set
        # e.g. 8901  -> 8 9 0 1
        # e.g. 8903 -> 8 9 0 3
        # if hf_bitstring = 1100000000, n_qubit = 10
        indices = _convert_arg_to_tensor_index(_arg, dim, tensor_rank=tensor_rank)
        if verbose:
            print(indices)

        # if tensor_order = 2,
        #    OK: (2, 0), (3, 0),...
        #    NG: (0, 0), (3, 3),...
        # if tensor_order = 4,
        #    OK: (2, 1, 0, 9), (2, 1, 0, 2), ...
        #    NG: (2, 0, 0, 2), (0, 1, 1, 0), ...
        if len(set(indices)) < tensor_rank // 2 + 1:
            _n += 1
            # continue
            None

        # (2, 1, 0, 9) and (1, 2, 0, 9) is EXPECTED to give identical fermionic terms
        # so we only count either
        elif not (sorted(indices) in sorted_inds_set):
            inds_set.append(tuple(indices))
            sorted_inds_set.append(sorted(indices))

            #if return_unscreened_term:
                #if len(indices) == 2:
                    #terms += [f"{indices[0]}^ {indices[1]}"]
                #elif len(indices) == 4:
                    #terms += [f"{indices[0]}^ {indices[1]}^ {indices[2]} {indices[3]}"]
                #elif len(indices) == 6:
                    #terms += [f"{indices[0]}^ {indices[1]}^ {indices[2]}^ {indices[3]} {indices[4]} {indices[5]}"]

            #else:
            terms += fermionic_excitation_term_from_tensorind(
                tuple(indices),
                hf_bitstring,
                sz_symmetry=sz_symmetry,
                refer_to_HF=refer_to_HF,
                molecule_type=molecule_type,
                impose_point_group_symmetry=impose_point_group_symmetry,
                mol=mol,
                act_inds=act_inds,
                groups=groups,
                characters=characters,
            )

            if n_term is not None:
                if len(terms) >= n_term:
                    terms = terms[:n_term]
                    break
            _n += 1
        else:
            _n += 1

    return terms

def _check_excitation_upperbound(term, excitation_upperbound, hf_bitstring):
    n_particle = hf_bitstring.count("1")
    bound = n_particle + excitation_upperbound
    return all([site < bound for site, action in term if action == 1])


def extract_k_body_hopping(
    operator,
    k,
    strictly_k_body=True,
):
    ret = type(operator)()
    for op in operator.get_operators():
        if op.many_body_order() != k * 2:
            continue
        if strictly_k_body:
            support = list(set([[s for s, _ in term] for term in op.terms][0]))
            if len(support) != 2 * k:
                continue
        ret += op
    return ret


def _generate_fermionic_excitation_terms_three_body(
    fermionic_hamiltonian: FermionOperator,
    hf_bitstring: str,
    n_set=20,
    refer_to_HF: bool = True,
    excitation_upperbound: int = 6,
):
    if excitation_upperbound is None:
        print("Setting excitation_upperbound = 6 for 3-body couplers...")
        excitation_upperbound = 6

    if n_set == 0:
        return []

    from openfermion import normal_ordered

    ham_sq = fermionic_hamiltonian * fermionic_hamiltonian
    ham_sq = normal_ordered(ham_sq)

    three_body = extract_k_body_hopping(
        ham_sq,
        k=3,
        strictly_k_body=True,
    )
    terms_tmp = three_body.terms

    terms_array = []
    coeff_array = []
    for term in terms_tmp:
        if refer_to_HF:
            if not _is_term_excitation_from_hartree_fork(term, hf_bitstring):
                continue
        if excitation_upperbound:
            if not _check_excitation_upperbound(term, excitation_upperbound, hf_bitstring):
                continue

        terms_array.append(term)
        coeff_array.append(terms_tmp[term])
    if len(terms_array) > 0:
        _, terms_array = zip(*sorted(zip(np.abs(coeff_array), terms_array))[::-1])
    return terms_array[:n_set]


def _convert_arg_to_tensor_index(arg, dim, tensor_rank):
    return [arg // (dim ** n) % dim for n in range(tensor_rank)[::-1]]


def histogram_of_indices(lis, dim):
    return np.histogram(lis, range=(0, dim - 1), bins=dim)[0].tolist()


def convert_arg_to_tensor_index(
    arguments,
    tensor_rank: int,
    dim: int,
):
    inds_set = []
    # histogram_set = []
    sorted_inds_set = []
    _n = 0

    while _n < len(arguments):
        _arg = arguments[_n]
        indices = _convert_arg_to_tensor_index(_arg, dim, tensor_rank=tensor_rank)

        # if tensor_order = 2,
        #    (2, 0)  -> OK
        #    (0, 0)  -> NG
        # if tensor_order = 4,
        #    (2, 1, 0, 9) -> OK
        #    (2, 1, 0, 2) -> OK
        #    (2, 0, 0, 2) -> NG
        if len(set(indices)) < tensor_rank // 2 + 1:
            _n += 1
            # continue
            None

        if not sorted(indices) in sorted_inds_set:
            inds_set.append(tuple(indices))
            sorted_inds_set.append(sorted(indices))
        # if not histogram_of_indices(indices, dim) in histogram_set:
        # inds_set.append(tuple(indices))
        # histogram_set.append(histogram_of_indices(indices, dim))

        _n += 1
    return inds_set


def _fermionic_excitation_term_simple(
    tensorind,
):
    rank = len(tensorind)
    return [
        " ".join(
            ["%d^" % i for i in tensorind[: rank // 2]] + ["%d" % i for i in tensorind[rank // 2 :]]
        )
    ]


def fermionic_excitation_term_from_tensorind(
    tensorind: tuple,
    hf_bitstring: str = None,
    sz_symmetry=True,
    point_group_symmetry=False,
    refer_to_HF=True,
    molecule_type=None,
    impose_point_group_symmetry=False,
    mol=None,
    act_inds=None,
    groups=None,
    characters=None,
):
    """
    returns fermionic excitation term (e.g., "2^ 3^ 1 0") used for virtual subspace expansion.
    tensorind: tuple, index of sites
    hf_bitstring: string
    """

    if hf_bitstring is None:
        return _fermionic_excitation_term_simple(tensorind)

    else:
        # raw term is generated as "0^ 2^", when tensorind = (0,2) and hf_bitstring = "11110000",
        # which simply considers the occupation of individual sites.
        _term_raw, creation_idx, annihilation_idx = _fermionic_excitation_term_raw(
            tensorind, hf_bitstring, return_idx=True
        )

        # Here, we transform the terms so that particle number is conserved.
        terms = []
        if len(creation_idx) == len(annihilation_idx):
            terms.append(_term_raw)

        # some indices are transferred to
        elif len(creation_idx) > len(annihilation_idx):
            dif = len(creation_idx) - len(annihilation_idx)
            for jdx in itertools.combinations(creation_idx, dif // 2):
                a_idx = annihilation_idx + list(jdx)
                c_idx = list(set(creation_idx) - set(jdx))

                # avoid double excitation like 8^ 9^ 1 1
                if len(a_idx) != len(set(a_idx)):
                    continue

                terms.append(_generate_term(c_idx, a_idx))

        elif len(annihilation_idx) > len(creation_idx):
            dif = len(annihilation_idx) - len(creation_idx)

            for jdx in itertools.combinations(annihilation_idx, dif // 2):
                c_idx = creation_idx + list(jdx)
                a_idx = list(set(annihilation_idx) - set(jdx))

                # avoid double excitation like 8^ 8^ 1 0
                if len(c_idx) != len(set(c_idx)):
                    continue

                terms.append(_generate_term(c_idx, a_idx))

        if sz_symmetry:
            terms = [_term for _term in terms if _is_term_sz_symmetry_preserving(_term)]

        if refer_to_HF:
            terms = [
                _term
                for _term in terms
                if _is_term_excitation_from_hartree_fork(_term, hf_bitstring)
            ]

        if impose_point_group_symmetry:
            if groups is None or characters is None:
                assert (
                    mol is not None and act_inds is not None
                ), " you need mol and act_inds to compute point group information."
                symmetry_type = mol.symmetry
                groups = get_irrep_group(mol, symmetry_type, act_inds)
                characters = get_character_table(symmetry_type)

            terms = [
                _term for _term in terms if __is_point_group_preserving(_term, groups, characters)
            ]

        return terms


def is_point_group_preserving(term, mol, act_inds):
    if type(term) == FermionOperator:
        assert len(term.terms) == 1
        term = str(term).split("[")[1].split("]")[0]

    symmetry_type = mol.symmetry

    groups = get_irrep_group(mol, symmetry_type, act_inds)
    characters = get_character_table(symmetry_type)
    return __is_point_group_preserving(term, groups, characters)


def get_irrep_group(mol, symmetry_type, act_inds, mf=None):
    if mf is None:
        if mol.spin == 0:
            mf = scf.RHF(mol).run(verbose=0)
            # mf.run(verbose = 0)
        else:
            mf = scf.ROHF(mol).run(verbose=0)

    irname = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff, check=False)

    mo_irrep_ids = [mol.irrep_id[mol.irrep_name.index(_ir)] for _ir in irname]
    mo_irrep_ids = [mo_irrep_ids[_ind] for _ind in act_inds]

    if symmetry_type == "C2v":
        groups = [[] for _ in range(4)]
    elif symmetry_type == "D2h":
        groups = [[] for _ in range(8)]
    elif symmetry_type == "Dooh" and set([mol.atom_symbol(i) for i in range(mol.natm)]) == {
        "Fe",
        "S",
    }:
        groups = [[] for _ in range(12)]
        d2h_ids = [0, 2, 3, 5, 6, 7, 10, 11, 14, 15, 16, 17]
        mo_irrep_ids = [d2h_ids.index(_id) for _id in mo_irrep_ids]

    else:
        raise Exception("symmetry_type = %s not implemented!" % symmetry_type)

    for mo_id, irrep_id in zip(range(len(mo_irrep_ids)), mo_irrep_ids):
        groups[irrep_id].append(2 * mo_id)
        groups[irrep_id].append(2 * mo_id + 1)
    return groups


def get_character_table(symmetry_type, mol=None):
    """
    Character table of D2h:
    http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=602&option=4
    (https://www.chegg.com/homework-help/questions-and-answers/n2o2-belongs-d2h-point-group-based-character-table-answer-following-questions-list-total-m-q11230661)
        |     E    |      C2(z)     |      C2(y)     |      C2(x)      |        i       |    \sig(xy)    |     \sig(xz)   |     \sig(yz)   |
    ag  |     1    |        1       |        1       |        1        |        1       |        1       |        1       |        1       |
    b1g |     1    |        1       |       -1       |       -1        |        1       |        1       |       -1       |       -1       |
    b2g |     1    |       -1       |        1       |       -1        |        1       |       -1       |        1       |       -1       |
    b3g |     1    |       -1       |       -1       |        1        |        1       |       -1       |       -1       |        1       |
    au  |     1    |        1       |        1       |        1        |       -1       |       -1       |       -1       |       -1       |
    b1u |     1    |        1       |       -1       |        -1       |        -1      |       -1       |        1       |        1       |
    b2u |     1    |        -1      |       1        |        -1       |        -1      |        1       |       -1       |        1       |
    b3u |     1    |        -1      |        -1      |        1        |        -1      |        1       |        1       |       -1       |


    C2v:
    #    |      E    |   C2    | \sig_v(xz) | \sig_v(yz)
    #a1|     1    |     1     |        1          |       1
    #a2|     1    |     1     |        -1         |       -1
    #b1|     1    |     -1    |        1          |       -1
    #b2|     1    |     -1    |        -1          |       1

    """
    if symmetry_type == "D2h":
        characters = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, -1, -1, 1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
            ]
        )
    elif symmetry_type == "C2v":
        characters = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    elif symmetry_type == "Dooh" and set([mol.atom_symbol(i) for i in range(mol.natm)]) == {
        "Fe",
        "S",
    }:

        # according to pyscf source code, we can relate Dooh and D2h
        # https://github.com/pyscf/pyscf/blob/master/pyscf/symm/basis.py
        # Dooh     ->  D2h
        # A1g   0      Ag    0
        # A2g   1      B1g   1
        # A1u   5      B1u   5
        # A2u   4      Au    4
        # E1gx  2      B2g   2
        # E1gy  3      B3g   3
        # E1ux  7      B3u   7
        # E1uy  6      B2u   6
        # E2gx  10     Ag    0
        # E2gy  11     B1g   1
        # E2ux  15     B1u   5
        # E2uy  14     Au    4
        # E3ux  17     B3u   7
        # E3uy  16     B2u   6

        # mol.irrep_id = [0, 2, 3, 5,
        #       6, 7, 10, 11,
        #       14, 15, 16, 17]
        # dooh = ["A1g", "E1gx", "E1gy", "A1u",
        #       "E1uy", "E1ux", "E2gx", "E2gy",
        #       "E2uy", "E2ux", "E3uy", "E3ux"]
        dooh_as_d2h_FeS2 = [
            "Ag",
            "B2g",
            "B3g",
            "B1u",
            "B2u",
            "B3u",
            "Ag",
            "B1g",
            "Au",
            "B1u",
            "B2u",
            "B3u",
        ]

        d2h = ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"]
        characters_d2h = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1],
                [1, -1, -1, 1, -1, -1, -1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1],
            ]
        )

        characters = np.array([characters_d2h[d2h.index(_c)] for _c in dooh_as_d2h_FeS2])

    else:
        raise Exception("symmetry_type = %s not implemented." % symmetry_type)
    return characters


def get_creation_annihilation_idx(term):
    if type(term) == str:
        creation_idx = [int(_s.split("^")[0]) for _s in term.split(" ") if "^" in _s]
        annihilation_idx = [int(_s) for _s in term.split(" ") if "^" not in _s]
    else:
        creation_idx = [_tup[0] for _tup in term if _tup[1] == 1]
        annihilation_idx = [_tup[0] for _tup in term if _tup[1] == 0]

    return creation_idx, annihilation_idx


def _is_term_sz_symmetry_preserving(term):
    creation_idx, anni_idx = get_creation_annihilation_idx(term)

    creation_op_spinsum = sum([i % 2 for i in creation_idx])
    anni_op_spinsum = sum([i % 2 for i in anni_idx])
    return creation_op_spinsum == anni_op_spinsum


def _is_term_excitation_from_hartree_fork(term, hf_bitstring):
    creation_idx, anni_idx = get_creation_annihilation_idx(term)
    is_empty_for_creation = all([hf_bitstring[idx] == "0" for idx in creation_idx])
    is_filled_for_anni = all([hf_bitstring[idx] == "1" for idx in anni_idx])

    return is_empty_for_creation and is_filled_for_anni


def __is_point_group_preserving(term, group, characters):
    if type(term) == FermionOperator:
        assert len(term.terms) == 1
        term = str(term).split("[")[1].split("]")[0]

    characters = np.array(characters)

    creation_idx, annihilation_idx = get_creation_annihilation_idx(term)
    creation_point_group_list = [which_point_group(idx, group) for idx in creation_idx]
    annihilation_point_group_list = [which_point_group(idx, group) for idx in annihilation_idx]

    if len(creation_point_group_list) == 0 and len(annihilation_point_group_list) == 0:
        return True
    elif len(creation_point_group_list) == 0:
        return all([i == 0 for i in annihilation_point_group_list])
    elif len(annihilation_point_group_list) == 0:
        return all([i == 0 for i in creation_point_group_list])
    else:
        creation_character = multiply_elementwise(
            *[
                characters[creation_point_group_list[i]]
                for i in range(len(creation_point_group_list))
            ]
        )
        annihilation_character = multiply_elementwise(
            *[
                characters[annihilation_point_group_list[i]]
                for i in range(len(annihilation_point_group_list))
            ]
        )
        assert len(creation_point_group_list) == len(
            annihilation_point_group_list
        ), "something is wrong, unexpected length of creation_point_group_list"

        return (creation_character == annihilation_character).all()


def multiply_elementwise(*arrays):
    n_array = len(arrays)
    ret = np.ones_like(arrays[0])
    for i in range(n_array):
        ret = np.multiply(ret, arrays[i])
    return ret


def which_point_group(idx, group):
    return [set([idx]).issubset(_g) for _g in group].index(True)


def _generate_term(creation_idx, annihilation_idx):
    return (
        " ".join(["%d^" % _idx for _idx in creation_idx])
        + " "
        + " ".join(["%d" % _idx for _idx in annihilation_idx])
    )


def _fermionic_excitation_term_raw(tensorind, hf_bitstring, return_idx=True):
    ind_unique, counts = np.unique(tensorind, return_counts=True)
    creation_idx = []
    annihilation_idx = []
    for n, _idx in enumerate(ind_unique):
        for _ in range(counts[n] // 2):
            creation_idx.append(_idx)
            annihilation_idx.append(_idx)

        if counts[n] % 2 == 1:
            if hf_bitstring[_idx] == "0":
                creation_idx.append(_idx)
            elif hf_bitstring[_idx] == "1":
                annihilation_idx.append(_idx)

    _raw_term = _generate_term(creation_idx, annihilation_idx)
    if not return_idx:
        return _raw_term
    else:
        return _raw_term, creation_idx, annihilation_idx


##########################################
# Extracting 1, 2-body terms from hamiltonian
##########################################

# applicable to both QubitOperator and FermionOperator
def get_operator_support(operator):
    supports = []
    if not operator.many_body_order() == 0:
        for key in list(operator.terms.keys()):
            for _tup in key:
                supports.append(_tup[0])
    return list(set(supports))


def get_n_qubit_from_openfermion(openfermion_object):
    """Returns the number of qubits present in openfermion QubitOperator object."""
    # vStr_term = str(openfermion_object).split(" +\n")
    # sites = []
    # for _str_term in vStr_term:
    # sites.append(_get_sites_from_str_term(_str_term))
    # return max(sum(sites, [])) + 1
    support = get_operator_support(openfermion_object)
    return len(support)


def mask_intrasubgraph_elements(A, partition):
    rank = len(A.shape)
    A_masked = np.copy(A)
    for _sites in partition:
        A_masked[np.ix_(*(_sites,) * rank)] = np.zeros_like(A[np.ix_(*(_sites,) * rank)])
    return A_masked


def extract_hopping_matrix(
    fermionic_hamiltonian: FermionOperator,
    dim=None,
):
    """
    extract t^{(1)}_{i,j} for t_ij c_i^\dag c_j
    """
    if dim is None:
        dim = get_n_qubit_from_openfermion(fermionic_hamiltonian)

    mat = np.zeros((dim, dim), dtype=complex)
    terms = fermionic_hamiltonian.terms
    for _term in terms:
        # each _term looks like
        # ((2, 1), (1, 0)), equivalent to FermionOperator(2^ 1)
        # ((2, 1), (2,0), (1, 1), (1, 0)), equivalent to FermionOperator(2^ 2 1^ 1)
        if len(_term) == 2 and (_term[0][1] == 1 and _term[1][1] == 0):
            i, j = _term[0][0], _term[1][0]
            mat[i, j] = terms[_term]

    return mat


# from openfermion_tools import get_n_qubit_from_openfermion


def extract_interaction_tensor(
    fermionic_hamiltonian: FermionOperator,
    dim=None,
    two_elec_hopping_only=False,
    refer_to_HF=False,
    hf_bitstring=None,
):
    """
    extract t^{(2)}_{i,j, k, l} for t_ijkl c_i^ c_j^ c_k c_l
    """
    if dim is None:
        dim = get_n_qubit_from_openfermion(fermionic_hamiltonian)

    assert_operator_order(fermionic_hamiltonian)

    tensor = np.zeros((dim, dim, dim, dim), dtype=complex)
    terms = fermionic_hamiltonian.terms
    for _term in terms:
        # each _term looks like
        # ((2, 1), (1, 0)), equivalent to FermionOperator(2^ 1)
        # ((2, 1), (2,0), (1, 1), (1, 0)), equivalent to FermionOperator(2^ 2 1^ 1)
        if len(_term) == 4 and (_term[0][1] == 1 and _term[1][1] == 1):
            i, j, k, l = _term[0][0], _term[1][0], _term[2][0], _term[3][0]

            if (two_elec_hopping_only) and len(set([i, j, k, l])) < 4:
                continue
            if refer_to_HF:
                assert hf_bitstring is not None, "we need HF configuration"
                if not sum([int(hf_bitstring[n]) for n in [i, j, k, l]]) == 2:
                    continue
            tensor[i, j, k, l] = terms[_term]

    return tensor


def assert_operator_order(fermionic_hamiltonian: FermionOperator):
    # raise Exception("to be written")
    terms = fermionic_hamiltonian.terms
    ordered_flags = []
    for _term in terms:
        if len(_term) == 2:
            ordered_flags.append(_term[0][1] == 1 and _term[1][1] == 0)
        elif len(_term) == 4:
            ordered_flags.append(
                _term[0][1] == 1 and _term[1][1] == 1 and _term[2][1] == 0 and _term[3][1] == 0
            )
    ordered = all(ordered_flags)
    if not ordered:
        raise Exception("Wrong fermionic operator order. must be c^ c^ c c or c^ c.")


##########################################
# Decompose into subgraph-wise operators
##########################################

cache_partition_dict = None
cache_partition = None


def which_subsystem(i, partition_sites):
    global cache_partition_dict
    global cache_partition
    if cache_partition_dict is None or cache_partition != partition_sites:
        cache_partition = partition_sites
        cache_partition_dict = {
            _site: i for i, _sites in enumerate(partition_sites) for _site in _sites
        }

    return cache_partition_dict[i]


def decompose_FermionOperator_partitionwise(
    operator: FermionOperator, 
    partition,
    return_coeff = False):
    assert (
        len(str(operator).split("+")) <= 1
    ), "currently only for identity or single Fermionic tensor product."
    string = str(operator)
    term = string[string.index("[") + 1 : string.index("]")]

    term_array = _split_term_partitionwise(term, partition)
    return [FermionOperator(_t) for _t in term_array]


def _split_term_partitionwise(term, partition):
    str_array = [[] for _ in partition]
    if term.split(" ") != [""]:
        for _s in term.split(" "):
            idx = int(re.sub(r"\D", "", _s))
            k = which_subsystem(idx, partition)
            jdx = [n == idx for n in partition[k]].index(True)
            str_array[k].append(str(jdx) + "^" * int("^" in _s))

    term_array = [" ".join(_t) for _t in str_array]
    return term_array


def FermionOperator_partitionwise(term: str, partition):
    term_array = _split_term_partitionwise(term, partition)
    return [FermionOperator(_t) for _t in term_array]


def get_sparse_operator_on_subgraph(operator_array, partition):
    assert len(operator_array) == len(partition)
    nqubit_array = [len(_sites) for _sites in partition]
    return [force_as_sparse(_op, n_qubits=_n) for _op, _n in zip(operator_array, nqubit_array)]


def force_as_sparse(operator, n_qubits=None):
    if isinstance(operator, scipy.sparse.csc_matrix):
        ret = operator
    else:
        ret = get_sparse_operator(operator, n_qubits=n_qubits)
    return ret


def get_qse_matrices_openfermion_partitionwise(vec0_array, Ppar, hamiltonian, partition):
    dim = len(Ppar)
    hsub = np.zeros((dim, dim), complex)
    ssub = np.zeros((dim, dim), complex)
    # n_qubit = int(np.log2(state.shape[0]))

    if isinstance(hamiltonian, scipy.sparse.csc_matrix):
        ham_sp = hamiltonian
    else:
        ham_sp = get_sparse_operator(hamiltonian)

    Ppar_sparse = [get_sparse_operator_on_subgraph(_Ppar, partition) for _Ppar in Ppar]
    Pvec_loc = [
        global_vector_from_local_operation(Ppar_sparse[i], vec0_array, partition)
        for i in range(len(Ppar_sparse))
    ]
    for i in range(dim):
        _Pivec = Pvec_loc[i]
        # _Pivec = tensor_product_over_subgraph(_Pivec_array, partition)
        for j in range(i, dim):
            _Pjvec = Pvec_loc[j]
            # _Pjvec = tensor_product_over_subgraph(_Pivec_array, partition)

            hsub[i, j] = _Pivec.conj() @ ham_sp @ _Pjvec
            ssub[i, j] = _Pivec.conj() @ _Pjvec

    hsub = hermitianize(hsub)
    ssub = hermitianize(ssub)
    return hsub, ssub


def global_vector_from_local_operation(local_operator_array, vec0_array, partition):
    _op = local_operator_array[0]
    assert not (
        isinstance(_op, QubitOperator) or isinstance(_op, FermionOperator)
    ), "convert so sparse operator first."
    vec_array = [_P_sp @ _vec0 for _P_sp, _vec0 in zip(local_operator_array, vec0_array)]

    from .local_solver import tensor_product_over_subgraph

    vec = tensor_product_over_subgraph(vec_array, partition)
    return vec


#########################################
# legacy (old or slow but correct)
#########################################


def fermionic_excitation_term_from_tensorind_old(
    tensorind: tuple,
    hf_bitstring: str = None,
    sz_symmetry=True,
    point_group_symmetry=False,
    refer_to_HF=True,
    molecule_type=None,
    impose_point_group_symmetry=False,
    #    impose_BeH2_point_group_symmetry = False,
    #    impose_H2O_point_group_symmetry = False,
    #    impose_N2_point_group_symmetry = False,
    #    impose_LiH_point_group_symmetry = False,
    # use_active_space=False
):
    """
    returns fermionic excitation term (e.g., "2^ 3^ 1 0") used for virtual subspace expansion.
    tensorind: tuple, index of sites
    hf_bitstring: string
    """

    if hf_bitstring is None:
        return _fermionic_excitation_term_simple(tensorind)

    else:
        # raw term is generated as "0^ 2^", when tensorind = (0,2) and hf_bitstring = "11110000",
        # which simply considers the occupation of individual sites.
        _term_raw, creation_idx, annihilation_idx = _fermionic_excitation_term_raw(
            tensorind, hf_bitstring, return_idx=True
        )

        # Here, we transform the terms so that particle number is conserved.
        terms = []
        if len(creation_idx) == len(annihilation_idx):
            terms.append(_term_raw)

        # some indices are transferred to
        elif len(creation_idx) > len(annihilation_idx):
            dif = len(creation_idx) - len(annihilation_idx)
            for jdx in itertools.combinations(creation_idx, dif // 2):
                a_idx = annihilation_idx + list(jdx)
                c_idx = list(set(creation_idx) - set(jdx))

                # avoid double excitation like 8^ 9^ 1 1
                if len(a_idx) != len(set(a_idx)):
                    continue

                terms.append(_generate_term(c_idx, a_idx))

        elif len(annihilation_idx) > len(creation_idx):
            dif = len(annihilation_idx) - len(creation_idx)

            for jdx in itertools.combinations(annihilation_idx, dif // 2):
                c_idx = creation_idx + list(jdx)
                a_idx = list(set(annihilation_idx) - set(jdx))

                # avoid double excitation like 8^ 8^ 1 0
                if len(c_idx) != len(set(c_idx)):
                    continue

                terms.append(_generate_term(c_idx, a_idx))

        if sz_symmetry:
            terms = [_term for _term in terms if _is_term_sz_symmetry_preserving(_term)]

        if refer_to_HF:
            terms = [
                _term
                for _term in terms
                if _is_term_excitation_from_hartree_fork(_term, hf_bitstring)
            ]

        if molecule_type == "BeH2" and impose_point_group_symmetry:
            use_active_space = False
            terms = [
                _term
                for _term in terms
                if _is_point_group_symmetry_preserving_BeH2_sto6g(_term, use_active_space)
            ]

        elif molecule_type == "LiH" and impose_point_group_symmetry:
            # use_active_space=False
            terms = [
                _term
                for _term in terms
                if _is_point_group_symmetry_preserving_LiH_sto6g(
                    _term,
                )
            ]

        elif molecule_type == "H2O" and impose_point_group_symmetry:
            if len(hf_bitstring) == 10:
                active_space_type = 0
            elif len(hf_bitstring) == 14:
                active_space_type = 1
            elif len(hf_bitstring) == 26:
                active_space_type = 2
            else:
                raise Exception("not familiar with this qubit number")
            terms = [
                _term
                for _term in terms
                if _is_point_group_symmetry_preserving_H2O_sto6g(_term, active_space_type)
            ]

        elif molecule_type == "N2" and impose_point_group_symmetry:
            if len(hf_bitstring) == 20:
                active_space_type = 0
            elif len(hf_bitstring) == 16:
                active_space_type = 1
            elif len(hf_bitstring) == 14:
                active_space_type = 2
            else:
                raise Exception("not familiar with this qubit number")

            terms = [
                _term
                for _term in terms
                if _is_point_group_symmetry_preserving_N2_sto6g(_term, active_space_type)
            ]

        elif impose_point_group_symmetry:
            raise Exception("not implemented yet")

        return terms


def _is_point_group_symmetry_preserving_N2_sto6g(term, active_space_type=0):
    """
    N2 : D_{\infty h}
    If we use D_2h for representation, as in
    https://chemistry.stackexchange.com/questions/91346/molecular-orbital-diagram-and-irreducible-representations-for-dinitrogen

    # type0 (20 qubit)
    Without active space:
    # sg (ag) = 0, 1, 4, 5, 12, 13
    # su (b1u)= 2, 3, 6, 7, 18, 19
    # pxu (b3u)= 8, 9
    # pyu (b2u)= 10, 11
    # pxg (b3g)= 14, 15
    # pyu (b2g)= 16, 17

    ## pxg (b3g)= 16, 17
    ## pyu (b2g)= 14, 15

    # type 1 (16 qubit)
    With active space freezing both 1s:
    # sg (ag)= 0, 1, 8, 9
    # su (b1u)= 2, 3, 14, 15
    # pxu (b3u)= 4, 5
    # pyu (b2u)= 6, 7
    # pxg (b3g)= 10, 11
    # pyu (b2g)= 12, 13

    # type 2
    With active space freezing both 1s:
    # sg (ag)= 0, 1, 6, 7
    # su (b1u)= 12, 13
    # pxu (b3u)= 2, 3
    # pyu (b2u)= 4, 5
    # pxg (b3g)= 8, 9
    # pyu (b2g)= 10, 11

    Character table of D2h:
    (https://www.chegg.com/homework-help/questions-and-answers/n2o2-belongs-d2h-point-group-based-character-table-answer-following-questions-list-total-m-q11230661)
        |     E    |      C2(z)     |      C2(y)     |      C2(x)      |        i       |    \sig(xy)    |     \sig(xz)   |     \sig(yz)   |
    ag  |     1    |        1       |        1       |        1        |        1       |        1       |        1       |        1       |
    b1g |     1    |        1       |       -1       |       -1        |        1       |        1       |       -1       |       -1       |
    b2g |     1    |       -1       |        1       |       -1        |        1       |       -1       |        1       |       -1       |
    b3g |     1    |       -1       |       -1       |        1        |        1       |       -1       |       -1       |        1       |
    au  |     1    |        1       |        1       |        1        |       -1       |       -1       |       -1       |       -1       |
    b1u |     1    |        1       |       -1       |        -1       |        -1      |       -1       |        1       |        1       |
    b2u |     1    |        -1      |       1        |        -1       |        -1      |        1       |       -1       |        1       |
    b3u |     1    |        -1      |        -1      |        1        |        -1      |        1       |        1       |       -1       |
    """

    # ag, b1u, b3u, b2u, b3g, b2g
    if active_space_type == 0:
        group = [[0, 1, 4, 5, 12, 13], [2, 3, 6, 7, 18, 19], [8, 9], [10, 11], [14, 15], [16, 17]]
    elif active_space_type == 1:
        group = [[0, 1, 8, 9], [2, 3, 14, 15], [4, 5], [6, 7], [10, 11], [12, 13]]
    elif active_space_type == 2:
        group = [[0, 1, 6, 7], [12, 13], [2, 3], [4, 5], [8, 9], [10, 11]]
    else:
        raise Exception("not implemented yet")

    # ag |     1, 1, 1, 1, 1, 1, 1, 1
    # b1u |     1, 1,-1,-1,-1,-1, 1, 1       |
    # b3u |     1,-1,-1, 1,-1, 1, 1,-1       |
    # b2u |     1,-1, 1,-1,-1, 1,-1, 1       |
    # b3g |     1,-1,-1, 1, 1,-1,-1, 1       |
    # b2g |     1,-1, 1,-1, 1,-1, 1,-1       |
    characters = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
        ]
    )
    return __is_point_group_preserving(term, group, characters)


def _is_point_group_symmetry_preserving_BeH2_sto6g(term, use_active_space=False):
    """
    BeH2 : D_{2h}
    following https://arxiv.org/abs/2109.02110

    Without active space:
    #  ag = 0, 1, 4, 5, 12, 13
    # b1u = 2, 3, 10, 11
    # b2u = 6, 7
    # b3u = 8, 9

    Character table:
    (https://www.chegg.com/homework-help/questions-and-answers/n2o2-belongs-d2h-point-group-based-character-table-answer-following-questions-list-total-m-q11230661)
        |     E    |      C2(z)     |      C2(y)     |      C2(x)      |        i       |    \sig(xy)    |     \sig(xz)   |     \sig(yz)   |
    ag  |     1    |        1       |        1       |        1        |        1       |        1       |        1       |        1       |
    b1g |     1    |        1       |       -1       |       -1        |        1       |        1       |       -1       |       -1       |
    b2g |     1    |       -1       |        1       |       -1        |        1       |       -1       |        1       |       -1       |
    b3g |     1    |       -1       |       -1       |        1        |        1       |       -1       |       -1       |        1       |
    au  |     1    |        1       |        1       |        1        |       -1       |       -1       |       -1       |       -1       |
    b1u |     1    |        1       |       -1       |        -1       |        -1      |       -1       |        1       |        1       |
    b2u |     1    |        -1      |       1        |        -1       |        -1      |        1       |       -1       |        1       |
    b3u |     1    |        -1      |        -1      |        1        |        -1      |        1       |        1       |       -1       |
    """
    assert not use_active_space, "not implemented."

    group = [[0, 1, 4, 5, 12, 13], [2, 3, 10, 11], [6, 7], [8, 9]]
    characters = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
        ]
    )

    return __is_point_group_preserving(term, group, characters)


def _is_point_group_symmetry_preserving_H2O_sto6g(term, active_space_type=0):
    """
    H2O: C_2v

    type 0:
    Without active space:
    # a1 = 0, 1, 2, 3, 6, 7, 10, 11
    # a2 = None
    # b1 = 8, 9,
    # b2 = 4, 5, 12, 13

    type1:
    With active space freezing 1s and 2pz:
    # a1 = 0, 1, 4, 5, 6, 7
    # a2 = None
    # b1 = None
    # b2 = 2, 3, 8, 9

    type 2:
    6-31G, without active space, 26 qubits
    a1 = 0, 1, 2, 3, 6, 7, 10, 11, 16, 17, 20, 21, 24, 25
    b1 = 8, 9, 18, 19
    b2 = 4, 5, 12, 13, 14, 15, 22, 23

    6-31G, with active space, 20 qubits
    a1 = [0, 1, 4, 5, 14, 15, 18, 19]?
    a2 = None
    b1 = [6, 7, 12, 13]?
    b2 = [2, 3, 8, 9, 10, 11,16, 17]?

    Character table:
        |      E    |   C2    | \sig_v(xz) | \sig_v(yz)
    a1|     1    |     1     |        1          |       1
    a2|     1    |     1     |        -1         |       -1
    b1|     1    |     -1    |        1          |       -1
    b2|     1    |     -1    |        -1          |       1
    """
    if type(term) == FermionOperator:
        assert len(term.terms) == 1
        term = str(term).split("[")[1].split("]")[0]

    if active_space_type == 0:
        group = [[0, 1, 4, 5, 6, 7], [], [], [2, 3, 8, 9]]
    elif active_space_type == 1:
        group = [[0, 1, 2, 3, 6, 7, 10, 11], [], [8, 9], [4, 5, 12, 13]]
    elif active_space_type == 2:
        group = [
            [0, 1, 2, 3, 6, 7, 10, 11, 16, 17, 20, 21, 24, 25],
            [],
            [8, 9, 18, 19],
            [4, 5, 12, 13, 14, 15, 22, 23],
        ]

    characters = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    return __is_point_group_preserving(term, group, characters)


def _is_point_group_symmetry_preserving_LiH_sto6g(term, active_space_type=0):
    """
    LiH: C_2v

    type 0:
    Without active space:
    # a1 = 0, 1, 2, 3, 4, 5, 10, 11
    # a2 = None
    # b1 = 6, 7
    # b2 = 8, 9

    Character table:
        |      E    |   C2    | \sig_v(xz) | \sig_v(yz)
    a1|     1    |     1     |        1          |       1
    a2|     1    |     1     |        -1         |       -1
    b1|     1    |     -1    |        1          |       -1
    b2|     1    |     -1    |        -1          |       1
    """
    if type(term) == FermionOperator:
        assert len(term.terms) == 1
        term = str(term).split("[")[1].split("]")[0]

    if active_space_type == 0:
        group = [[0, 1, 2, 3, 4, 5, 10, 11], [], [6, 7], [8, 9]]

    characters = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
    return __is_point_group_preserving(term, group, characters)


def _is_point_group_symmetry_preserving_BeH2_sto6g_old(term, use_active_space=False):
    """
    #  ag = 0, 1, 4, 5, 12, 13
    # b1u = 2, 3, 10, 11
    # b2u = 6, 7
    # b3u = 8, 9
    """
    assert not use_active_space, "not implemented."

    group = [[0, 1, 4, 5, 12, 13], [2, 3, 10, 11], [6, 7], [8, 9]]
    c_idx = [int(_s.split("^")[0]) for _s in term.split(" ") if "^" in _s]
    a_idx = [int(_s) for _s in term.split(" ") if "^" not in _s]

    assert len(c_idx) == 2 and len(a_idx) == 2, "2-elec excitation for now."

    c_pg = [which_point_group(idx, group) for idx in c_idx]
    a_pg = [which_point_group(idx, group) for idx in a_idx]

    # PG symmetric case 1: double excitation from identical spin orbital
    # e.g. "10^ 11^ 4 5" (ag, ag) → (b1u, b1u)
    flag1 = c_pg[0] == c_pg[1] and a_pg[0] == a_pg[1]

    # PG symmetric case 2:
    # e.g. "10^ 12^ 0 2" (ag, b1u) → (ag, b1u)
    flag2 = set(c_pg) == set(a_pg)

    return flag1 or flag2


def _is_point_group_symmetry_preserving_H2O_sto6g_old(term, use_active_space=True):
    """
    Without active space:
    # a1 = 0, 1, 2, 3, 6, 7, 10, 11
    # b1 = 8, 9,
    # b2 = 4, 5, 12, 13
    With active space freezing 1s and 2pz:
    # a1 = 0, 1, 4, 5, 6, 7
    # b1 = None
    # b2 = 2, 3, 8, 9
    """

    if type(term) == FermionOperator:
        assert len(term.terms) == 1
        term = str(term).split("[")[1].split("]")[0]

    if use_active_space:
        group = [[0, 1, 4, 5, 6, 7], [2, 3, 8, 9]]
    else:
        group = [[0, 1, 2, 3, 6, 7, 10, 11], [8, 9], [4, 5, 12, 13]]

    c_idx = [int(_s.split("^")[0]) for _s in term.split(" ") if "^" in _s]
    a_idx = [int(_s) for _s in term.split(" ") if "^" not in _s]

    assert len(c_idx) == 2 and len(a_idx) == 2, "2-elec excitation for now."

    c_pg = [which_point_group(idx, group) for idx in c_idx]
    a_pg = [which_point_group(idx, group) for idx in a_idx]

    # PG symmetric case 1: double excitation from identical spin orbital
    # e.g. "8^ 9^ 0, 1" (ag, ag) → (b2, b2)
    flag1 = c_pg[0] == c_pg[1] and a_pg[0] == a_pg[1]

    # PG symmetric case 2:
    # e.g. "6^ 8^ 0 2" (ag, b1u) → (ag, b1u)
    flag2 = set(c_pg) == set(a_pg)

    return flag1 or flag2


