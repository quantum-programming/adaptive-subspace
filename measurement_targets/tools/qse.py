from typing import List, Union

import numpy as np
import qutip as qt
import scipy
from openfermion import get_sparse_operator, hermitian_conjugated
from openfermion.ops import FermionOperator, QubitOperator
from qiskit.opflow import PauliSumOp
from qiskit_nature.operators.second_quantization import FermionicOp

import vse.qiskit_utils  # this introduces ".from_openfermion"


def canonical_chop(A, eps=1e-8, drop_negative=True, std_A=None, replace_zero = False):
    B = np.copy(A)
    v, u = np.linalg.eigh(B)
    v_round = chop(v, eps=eps, drop_negative=drop_negative, replace_zero = replace_zero)
    A_chop = u @ np.diag(v_round) @ u.conj().T
    if std_A is not None:
        stdv = (u.conj().T @ std_A @ u).diagonal()
        stdv_round = stdv * (v_round / v)
        stdA_chop = u @ np.diag(stdv_round) @ u.conj().T
        return A_chop, stdA_chop
    else:
        return A_chop


def chop(A, eps=1e-5, drop_negative=True, replace_zero = False):
    B = np.copy(A)
    if not drop_negative:
        B[np.abs(A) < eps] = 0
    else:
        if replace_zero:
            B[A <eps] = 0
        else:
            B[A < eps] = eps
    return B


def hermitianize(mat):
    if np.abs(np.triu(mat) - np.tril(mat).conj().T).sum() > 1e-5:
        if np.abs(np.triu(mat)).sum() > np.abs(np.tril(mat)).sum():
            return np.triu(mat) + np.triu(mat).conj().T - np.diag(mat.diagonal())
        else:
            return np.tril(mat) + np.tril(mat).conj().T - np.diag(mat.diagonal())

    else:
        # do nothing if it is already hermitian
        return mat


def get_partial_matrix(mat, args):
    return np.copy(mat)[:, args][args, :]


def solve_qse_equation(
    Hsub,
    Ssub,
    args=None,
    eps=1e-8,
    verbose=0,
    norm_min=1e-5,
    norm_max=1e8,
    reference_energy = None,
    atol = 10,
    return_state=False,
    psi0=None,
    P=None,
    return_alpha=False,
):

    if return_state:
        assert psi0 is not None
        assert P is not None

    if args is not None:
        Hsub = get_partial_matrix(Hsub, args)
        Ssub = get_partial_matrix(Ssub, args)
    else:
        args = list(range(Hsub.shape[0]))


    dim = Hsub.shape[0]
    # Ssub = canonical_chop(Ssub, eps=eps) + np.diag(np.ones(dim)) * eps
    Ssub = canonical_chop(Ssub, eps=eps)

    # _v, _vecs = scipy.linalg.eig(Hsub, b=Ssub)
    _v, _vecs = scipy.linalg.eigh(Hsub, b=Ssub)

    try_again = True
    n = 0
    arg_sorted = _v.real.argsort()
    while try_again:
        alphas = _vecs[:, arg_sorted[n]]

        _vecnorm = np.linalg.norm(alphas)
        _norm = alphas.conj() @ Ssub @ alphas
        _en = (alphas.conj() @ Hsub @ alphas) / _norm

        if verbose >= 1:
            print("n=%d, en=%.5f (%.5f), norm=%.5f" % (n, _en, _v.real[arg_sorted[n]], _norm))
        n += 1
        if np.abs(_norm) > norm_min and _vecnorm < norm_max:
            if reference_energy is not None and np.abs(_en - reference_energy) > atol:
                continue
            try_again = False

    en = _en

    if return_state:
        n_qubit = int(np.log2(psi0.shape[0]))
        state = sum(
            [alphas[i] * force_as_sparse(P[args[i]], n_qubits=n_qubit) * psi0 for i in range(dim)]
        )
        # state = sum([alphas[i] * get_sparse_operator(P[i], n_qubits = n_qubit) * psi0 for i in args])
        return en.real, state / (np.linalg.norm(state))
    elif return_alpha:
        return en.real, alphas
    else:
        return en.real


def solve_noisy_qse_equation(
    Hsub, 
    Ssub, 
    args  = None,
    eps = 1e-8, 
    norm_max = 100, 
    atol_from_h00 = 10, 
    return_alpha = False, 
    verbose = 0
):
    
    if args is not None:
        Hsub = get_partial_matrix(Hsub, args)
        Ssub = get_partial_matrix(Ssub, args)
    else:
        args = list(range(Hsub.shape[0]))    

    dim = Hsub.shape[0]
    #Ssub_ = canonical_chop(Ssub, eps=eps)
    Ssub_ = canonical_chop(Ssub, eps=eps) + np.diag(np.ones(dim)) * eps

    #_v, _vecs = scipy.linalg.eig(Hsub, b=Ssub)
    _v, _vecs = scipy.linalg.eigh(Hsub, b=Ssub_)

    #try_again = True
    #n = 0
    arg_sorted = _v.real.argsort()
    narg = _v.shape[0]
    for n in range(narg):
        alphas = _vecs[:, arg_sorted[n]]

        _vecnorm = np.linalg.norm(alphas)
        _en = _v[arg_sorted[n]]

        if verbose >= 1:
            print("n=%d, en=%.5f (%.5f), vecnorm=%.5f" % (n, _en, _v.real[arg_sorted[n]], _vecnorm))
        #n += 1
        if np.abs(_en - Hsub[0, 0]) < atol_from_h00 and _vecnorm < norm_max:
            en = _en
            break

    if n == narg-1:
        en = np.NaN
    
    
    if return_alpha:
        return en.real, alphas
    else:
        return en.real
            


def force_as_sparse(
    operator: Union[PauliSumOp, FermionOperator, QubitOperator, scipy.sparse.csc_matrix],
    n_qubits=None,
):
    if isinstance(operator, scipy.sparse.csc_matrix):
        # already scipy.sparse
        ret = operator
    elif isinstance(operator, FermionOperator) or isinstance(operator, QubitOperator):
        # openfermion operator
        ret = get_sparse_operator(operator, n_qubits=n_qubits)
    elif isinstance(operator, PauliSumOp):
        # qiskit operator
        ret = operator.to_spmatrix()
    return ret


def compute_qse_matrices(
    state: np.ndarray,
    P: List[Union[PauliSumOp, FermionOperator, QubitOperator]],
    hamiltonian: Union[PauliSumOp, FermionOperator, QubitOperator],
    use_qse_operator_explicit=False,
):
    dim = len(P)
    hsub = np.zeros((dim, dim), complex)
    ssub = np.zeros((dim, dim), complex)
    n_qubit = int(np.log2(state.shape[0]))

    ham_sp = force_as_sparse(hamiltonian, n_qubits=n_qubit)
    P_sparse = [force_as_sparse(P[j], n_qubits=n_qubit) for j in range(dim)]
    Pdag_sparse = [_Pj.conj().T for _Pj in P_sparse]

    if not use_qse_operator_explicit:
        # This is faster in classical siumlation

        Pv_loc = [_Pi_sp * state for _Pi_sp in P_sparse]

        for i in range(dim):
            for j in range(i, dim):
                # Pj_v =
                hsub[i, j] = Pv_loc[i].conj() @ ham_sp @ Pv_loc[j]
                ssub[i, j] = Pv_loc[i].conj() @ Pv_loc[j]

    else:
        # In actual quantum simulation,
        # we need to evaluate Pj^dag H Pi or Pj^dag Pi
        for i in range(dim):
            _Pidag = Pdag_sparse[i]
            for j in range(i, dim):
                _Pj = P_sparse[j]

                # multiplication * is OK for sparse matrix
                hij = _Pidag * ham_sp * _Pj
                sij = _Pidag * _Pj

                hsub[i, j] = expectation(hij, state)
                ssub[i, j] = expectation(sij, state)

    hsub = hermitianize(hsub)
    ssub = hermitianize(ssub)
    return hsub, ssub


def compute_qse_observable(
    alphas, 
    osub, 
    ssub, 
    args = None
    ):
    def get_partial_matrix(mat, args):
        return np.copy(mat)[:, args][args, :]

    if args is not None:
        osub = get_partial_matrix(osub, args)
        ssub = get_partial_matrix(ssub, args)

    return (alphas.conj() @ osub @ alphas)/(alphas.conj() @ ssub @ alphas)

##############################################
# legacy
##############################################


def solve_qse_openfermion(
    state,
    P,
    hamiltonian,
    return_state=False,
    eps=1e-8,
    norm_min=1e-5,
    use_qse_operator_explicit=False,
):
    h0, s0 = get_qse_matrices_openfermion(
        state, P, hamiltonian, use_qse_operator_explicit=use_qse_operator_explicit
    )
    ret = solve_qse_equation(
        h0,
        s0,
        return_state=return_state,
        P=P,
        psi0=state,
        eps=eps,
        norm_min=norm_min,
    )

    return ret
    # if return_state:
    # return en, vec
    # else:
    # return en


def get_qse_matrices_openfermion(state, P, hamiltonian, use_qse_operator_explicit=False):
    dim = len(P)
    hsub = np.zeros((dim, dim), complex)
    ssub = np.zeros((dim, dim), complex)
    n_qubit = int(np.log2(state.shape[0]))

    ham_sp = force_as_sparse(hamiltonian, n_qubits=n_qubit)

    if not use_qse_operator_explicit:
        # This is faster in classical siumlation
        P_sparse = [force_as_sparse(P[j], n_qubits=n_qubit) for j in range(dim)]
        Pv_loc = [_Pi_sp * state for _Pi_sp in P_sparse]

        for i in range(dim):
            for j in range(i, dim):
                # Pj_v =
                hsub[i, j] = Pv_loc[i].conj() @ ham_sp @ Pv_loc[j]
                ssub[i, j] = Pv_loc[i].conj() @ Pv_loc[j]

    else:
        # In actual quantum simulation,
        # we need to evaluate Pj^dag H Pi or Pj^dag Pi
        P_sparse = [get_sparse_operator(P[j], n_qubits=n_qubit) for j in range(dim)]
        Pdag_sparse = [_Pj.conj().T for _Pj in P_sparse]
        for i in range(dim):
            _Pidag = Pdag_sparse[i]
            for j in range(i, dim):
                _Pj = P_sparse[j]

                # multiplication * is OK for sparse matrix
                hij = _Pidag * ham_sp * _Pj
                sij = _Pidag * _Pj

                hsub[i, j] = expectation(hij, state)
                ssub[i, j] = expectation(sij, state)

    hsub = hermitianize(hsub)
    ssub = hermitianize(ssub)
    return hsub, ssub


def __get_qse_matrices_openfermion_slow(
    state,
    P,
    hamiltonian,
):
    dim = len(P)
    hsub = np.zeros((dim, dim), complex)
    ssub = np.zeros((dim, dim), complex)
    n_qubit = int(np.log2(state.shape[0]))

    ham_sp = get_sparse_operator(hamiltonian, n_qubits=n_qubit)
    for i in range(dim):
        for j in range(i, dim):
            _Pidag = get_sparse_operator(hermitian_conjugated(P[i]), n_qubits=n_qubit)
            _Pj = get_sparse_operator(P[j], n_qubits=n_qubit)
            hij = _Pidag * ham_sp * _Pj
            sij = _Pidag * _Pj

            hsub[i, j] = expectation(hij, state)
            ssub[i, j] = expectation(sij, state)
    hsub = hermitianize(hsub)
    ssub = hermitianize(ssub)
    return hsub, ssub


def get_qse_matrices_qutip(
    state,
    P,
    hamiltonian,
):

    dim = len(P)
    Hsub = np.zeros((dim, dim), complex)
    Ssub = np.zeros((dim, dim), complex)

    for i in range(dim):
        for j in range(i, dim):
            _Pidag = P[i].dag()
            _Pj = P[j]
            hij = _Pidag * hamiltonian * _Pj
            sij = _Pidag * _Pj

            Hsub[i, j] = qt.expect(hij, state)
            Ssub[i, j] = qt.expect(sij, state)

    Hsub = hermitianize(Hsub)
    Ssub = hermitianize(Ssub)
    return Hsub, Ssub


def get_qse_solution_qutip(state, P, hamiltonian, eps=1e-8, norm_min=1e-5):
    if state.isket:
        state = qt.ket2dm(state)
    Hsub, Ssub = get_qse_matrices_qutip(state, P, hamiltonian)
    Ssub = canonical_chop(
        Ssub,
        eps=eps,
    )
    _v, _vecs = scipy.linalg.eig(Hsub, b=Ssub)

    dim = Hsub.shape[0]
    if state is not None:
        alphas = _vecs[:, np.argmin(_v)]
        tmp = sum([alphas[i] * P[i] for i in range(dim)])
        rho_tmp = tmp * state * tmp.dag()

        rho_qse = rho_tmp / (rho_tmp.tr())
        return np.min(_v), rho_qse
    else:
        return np.min(_v)


def get_largest_eigvalstate(rho):
    _vals, _vecs = rho.eigenstates()
    return _vecs[np.argmax(_vals)]


def expectation(operator, state):
    if str(type(operator)).split(".")[0][8:] == "openfermion":
        n_qubit = int(np.log2(state.shape[0]))
        operator = get_sparse_operator(operator, n_qubits=n_qubit)
    if len(state.shape) == 1:
        return state.conj() @ operator @ state
    elif len(state.shape) == 2:
        return np.trace(operator @ state)
