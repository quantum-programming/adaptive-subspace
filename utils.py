from ast import operator
from typing import Tuple, Union

from openfermion import QubitOperator, utils

# from create_pauli import create_pauli_id_from_openfermion


hamiltonian_names = ["h2_4bk.data", "h2_8bk.data", "LiH_12bk.data", "BeH2_14bk.data"]


def if_single_pauli_product(operator):
    return len(str(operator).split("+")) == 1


def create_pauli_id_from_openfermion(operator: Union[QubitOperator, Tuple], n_qubit):
    if type(operator) == QubitOperator:
        assert if_single_pauli_product(operator), "input is not single product of pauli."
        terms = list(operator.terms.keys())[0]
    elif type(operator) == tuple:
        terms = operator
        
    qidx_list = []
    pauli_id_list = []
    plist = ["I", "X", "Y", "Z"]
    for _term in terms:
        qidx_list.append(_term[0])
        pauli_id_list.append(plist.index(_term[1]))    
    
    ret = [0]*n_qubit
    for _qid, _pid in zip(qidx_list, pauli_id_list):
        ret[_qid] = _pid
    return ret 


# 0:h2_4bk 1:h2_8bk 2:LiH_12bk 3:BeH2_14bk(とりあえず)
def hamiltonian_to_list(type):
    ham = utils.load_operator(
        file_name=hamiltonian_names[type],
        data_directory="Hamiltonian",
        plain_text=True,
    )
    import numpy as np
    import scipy
    from openfermion import count_qubits, get_sparse_operator

    n_qubit = count_qubits(ham)
    ham_sp = get_sparse_operator(ham)
    vals, vecs = scipy.sparse.linalg.eigsh(ham_sp, which="SA", k=10)
    gsvec = vecs[:, 0]
    print(f"ground state energy = {vals[0]:.5f}")

    ret = []
    wei = []
    for op in str(ham).split("+"):
        wei.append(float(op.split(" ")[0]))
        ret.append(QubitOperator(op))
    return (gsvec, ret, wei, n_qubit)
