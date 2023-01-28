# Keep h5py deprecation warning off
import warnings

warnings.filterwarnings("ignore")

import argparse
import itertools
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import openfermion
import scipy

import numpy as np
import glob

def prints(text):
    #global rank
    #if rank == 0:
    print(text)

from openfermion.ops import FermionOperator
from openfermion import (
    get_fermion_operator,
)

from pyscf import mp, scf
from scipy import linalg as LA


sys.path.append("./tools")
from general_active_space import get_molecular_hamiltonian_generalAS

from molecule_object import (
    build_molecule,
    get_BeH2_object,
    get_H2O_object,
    get_LiH_object,
    get_N2_object,
)

from virtual_subspace_fermion import (
    generate_fermionic_excitation_terms,
    get_character_table,
    get_irrep_group,
)

# mean-field calculation to construct Hamiltonian
def scf_calculation(mol, irrep_nelec=None, dm0=None):

    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)

    if irrep_nelec is not None:
        mf.irrep_nelec = irrep_nelec
    mf = scf.newton(mf)
    EHF = mf.kernel(dm0=dm0)
    a = mf.stability()[0]
    EHF = mf.kernel(mf.make_rdm1(a, mf.mo_occ))
    mf.stability()
    mf.analyze()
    return mf, EHF


# Needed for transformation into natural orbital
def compute_natural_orbitals(mf, mp2_solver):

    rho = mp2_solver.make_rdm1()
    if isinstance(rho, tuple):
        rho = rho[0] + rho[1]
    nat_occ, nat_orb = LA.eigh(-rho)
    nat_occ = -nat_occ
    return np.dot(mf.mo_coeff, nat_orb)



#molecule_type = "H2O"
#basis_type = "sto-6g"
orbital_type = "molecule"

nf = 0
spin = 0
occ_inds = list(range(nf))

parser = argparse.ArgumentParser(description="generate H2O molecule Hamiltonian")
parser.add_argument("--molecule_type", type=str, default="H2O", help="molecule type")
parser.add_argument("--basis_type", type=str, default="sto-6g", help="basis type")

args = parser.parse_args()
molecule_type = args.molecule_type
basis_type = args.basis_type
#basis_type = args.get("basis_type", "sto-6g")


filename_list = glob.glob(f"./config/setup_{molecule_type}_{basis_type}*")
for _f in filename_list:
    setupfile = json.load(open(_f))
    if setupfile["basis_type"] != basis_type:
        continue

    for key, value in setupfile.items():
        args.__setattr__(key, value)

    L=args.L


prints("\n======== system setup =========")
prints("molecule type = %s" % molecule_type)
prints("basis = %s" % basis_type)
prints("L = %.5f" % L)
prints("==================================================\n\n")


##############################################################
# Building molecule object
##############################################################
# MolecularData: openfermion object
prints("\n\n==================================================")
prints("Building molecule...")
prints("==================================================\n\n")

t0 = time.time()
mol = build_molecule(molecule_type, basis_type, L=L, spin=spin)
if molecule_type == "FeS2":
    # active space
    assert nf is not None, "determine active space size. nf=19 for CAS"
    assert spin == 6, "spin not set correctly, take spin=6 for FeS2 simulation"
    irrep_nelec = {
        "A1g": (9, 8),
        "A1u": (7, 6),
        "E1ux": (4, 4),
        "E1uy": (4, 4),
        "E1gx": (3, 2),
        "E1gy": (3, 2),
        "E2gx": (1, 0),
        "E2gy": (1, 0),
    }

    # mf, EHF = run_scf_calculation(mol, irrep_nelec)
    mf, EHF = scf_calculation(mol, irrep_nelec, dm0=None)
    # print("Trial %d-1, rank=%d, EHF = %.8f"%(_, rank, EHF))
    dm0 = mf.make_rdm1()
    mf, EHF = scf_calculation(mol, irrep_nelec, dm0=None)

    prints("EHF = %.8f" % EHF)
else:
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.run()
    EHF = mf.e_tot
t_molbuild = time.time() - t0
prints("...done in %.5f sec." % t_molbuild)

if orbital_type == "natural":

    mymp = mp.MP2(mf, frozen=occ_inds)
    mf.mo_coeff = compute_natural_orbitals(mf, mymp)
    # raise Exception("to be written")


n_orb = mol.nao
# act_inds = list(set(range(n_orb)) - set(occ_inds))
act_inds = sorted(list(set(range(n_orb)) - set(occ_inds)))
n_qubit = 2 * len(act_inds)


symmetry_type = mol.symmetry
groups = get_irrep_group(mol, symmetry_type, act_inds, mf=mf)
characters = get_character_table(symmetry_type, mol=mol)

import os, sys

directory = f"{molecule_type}_{basis_type}_r_{L:.5f}_{n_qubit}qubits"
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)


##############################################################
# Build fermionic Hamiltonian
##############################################################

prints("\n\n==================================================")
prints("generating Fermionic Hamiltonian...")
prints("==================================================\n\n")

occupied_spinorbitals = sum([[2 * i, 2 * i + 1] for i in occ_inds], [])
active_spinorbitals = sum([[2 * i, 2 * i + 1] for i in act_inds], [])
t0 = time.time()
molecular_ham = get_molecular_hamiltonian_generalAS(
    mol, occupied_spinorbitals, active_spinorbitals, pyscf_mf=mf
)
# molecular_ham = molecule.get_molecular_hamiltonian(occ_inds, act_inds)
ham_f = get_fermion_operator(molecular_ham)    

prints("...done.")

from openfermion import jordan_wigner, bravyi_kitaev

print("transforming into qubit hamiltonian...")
ham_jw = jordan_wigner(ham_f)
ham_bk = bravyi_kitaev(ham_f)
print("...done.")


openfermion.save_operator(ham_f, file_name="fermionic_hamiltonian", data_directory = directory, allow_overwrite=True)
openfermion.save_operator(ham_jw, file_name="jw_hamiltonian", data_directory = directory,allow_overwrite=True)
openfermion.save_operator(ham_bk, file_name="bk_hamiltonian", data_directory = directory,allow_overwrite=True)
print("...saved.")