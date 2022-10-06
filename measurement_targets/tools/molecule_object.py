try:
    # this worked for openfermion v0.10.0, not for v1.1.0
    from openfermion.hamiltonians import MolecularData
except:
    # this worked for openfermion v1.1.0
    from openfermion import MolecularData

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def assert_directory_exists(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def build_molecule(molecule_type, basis_type, L, spin=0, charge = 0):
    mol = gto.Mole()

    if molecule_type == "H2O":
        degree = 104.478
        rad = 2 * np.pi * (degree/360)
        mol.atom = "O 0 0 0; H %.5f 0 0; H %.5f %.5f 0"%(L, L*np.cos(rad), L*np.sin(rad))
        mol.symmetry = "C2v"

    elif molecule_type == "N2":
        mol.atom = "N 0 0 0; N 0 0 %.5f"%(L)
        mol.symmetry = "D2h"

    elif molecule_type == "C2":
        mol.atom = "C 0 0 0; C 0 0 %.5f"%(L)
        mol.symmetry = "D2h"

    elif molecule_type == "LiH":
        mol.atom = "Li 0 0 0; H 0 0 %.5f"%(L)
        mol.symmetry = "C2v"

    elif molecule_type == "BeH2":
        mol.atom = "Be 0 0 0; H 0 0 %.5f; H 0 0 %.5f"%(L, -L)
        mol.symmetry = "C2v"

    elif molecule_type == "Cr2":
        mol.atom = "Cr 0 0 0; Cr 0 0 %.5f"%(L)
        mol.symmetry = "D2h"

    elif molecule_type == "FeS2":
        mol.atom = "Fe 0 0 0; S 0 0 %.5f; S 0 0 %.5f"%(L, -L)
        mol.symmetry = "Dooh"
        assert spin!=0, "spin is expected to be nonzero for ground state of FeS2."

    elif molecule_type == "H2S":
        degree = 92.11 # according to CCCBDB
        rad = 2 * np.pi * (degree/360)
        mol.atom = "S 0 0 0; H %.5f 0 0; H %.5f %.5f 0"%(L, L*np.cos(rad), L*np.sin(rad))
        mol.symmetry = "C2v"

    elif molecule_type == "HF":
        mol.atom = "H 0 0 0; F 0 0 %.5f"%(L)
        mol.symmetry = "C2v"

    elif molecule_type == "HCl":
        mol.atom = "H 0 0 0; Cl 0 0 %.5f"%(L)
        mol.symmetry = "C2v"        

    elif molecule_type in ["H%d"%i for i in range(100)]:
        n_H = int(molecule_type[1:])
        mol= _build_Hn(basis_type, L, n_H)    
        
    else:
        raise Exception("not implemented.")

    mol.basis = basis_type
    mol.spin = spin
    mol.charge = charge
    mol.build()
    return mol



from pyscf import gto
def _build_H2O(basis_type, L, degree = 104.478):
    rad = 2 * np.pi * (degree/360)
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H %.5f 0 0; H %.5f %.5f 0"%(L, L*np.cos(rad), L*np.sin(rad))
    #geometry = [ ['O', [0,  0, 0]], ['H', [r,  0, 0]], ["H", [r*np.cos(rad), r*np.sin(rad), 0]]] 
    mol.symmetry = "C2v"
    mol.basis = basis_type
    #mol.build()
    return mol

def _build_N2(basis_type, L,):
    mol = gto.Mole()
    #mol1.atom = 'Be 0 0 0; H 0 0 %.5f; H 0 0 %.5f'%(r, -r)
    mol.atom = "N 0 0 0; N 0 0 %.5f"%(L)
    mol.symmetry = "D2h"
    mol.basis = basis_type
    #mol.build()
    return mol

def _build_LiH(basis_type, L,):
    mol = gto.Mole()
    mol.atom = "Li 0 0 0; H 0 0 %.5f"%(L)
    mol.symmetry = "C2v"
    mol.basis = basis_type
    #mol.build()

    return mol   

def _build_BeH2(basis_type, L,):
    mol = gto.Mole()
    mol.atom = "Be 0 0 0; H 0 0 %.5f; H 0 0 %.5f"%(L, -L)
    mol.symmetry = "C2v"
    mol.basis = basis_type
    #mol.build()

    return mol         

def _build_Cr2(basis_type, L,):
    mol = gto.Mole()
    mol.atom = "Cr 0 0 0; Cr 0 0 %.5f"%(L)
    mol.symmetry = "D2h"
    mol.basis = basis_type
    #mol.build()

    return mol    

def _build_Hn(basis_type, L, n_H):
    mol = gto.Mole()
    atom_str = "H 0 0 0"
    for i in range(1, n_H):
        atom_str += "; H 0 0 %.5f"%(L * i)
    mol.atom = atom_str
    mol.symmetry = "D2h"
    mol.basis = basis_type
    #mol.build()
    return mol


def get_H_object(r, 
                n_H, 
                multiplicity = 1, 
                charge = 0, 
                basis_type = "sto-3g",):
    geometry = [ ['H', [n * r,  0, 0]] for n in range(n_H)] 
    molecule_type = "H%d"%n_H
    
    #filename = "molecules/" + molecule_type + "_" + basis_type + "/" + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)
    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    filename = directory + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)

    molecule = MolecularData(geometry, basis_type, multiplicity , charge, 
                                filename=filename,)
    if rank == 0:
        molecule.save()
    return molecule

def get_LiH_object(r, multiplicity = 1, charge = 0, basis_type = "sto-3g"):
    geometry = [ ['Li', [0,  0, 0]], ['H', [r,  0, 0]]] 
    molecule_type = "LiH"
    
    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    filename = directory + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)

    molecule = MolecularData(geometry, basis_type, multiplicity , charge, 
                                filename=filename,)
    if rank == 0:
        molecule.save()
    return molecule

def get_H2O_object(r, 
                    degree = 104.478, 
                    multiplicity = 1, 
                    charge = 0, 
                    basis_type = "sto-3g",
                    rank = 0
                    ):
    rad = 2*np.pi * (degree/360)
    geometry = [ ['O', [0,  0, 0]], ['H', [r,  0, 0]], ["H", [r*np.cos(rad), r*np.sin(rad), 0]]] 
    
    molecule_type = "H2O"

    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    filename = directory + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)

    molecule = MolecularData(geometry, basis_type, multiplicity , charge, 
                                filename=filename,)
    if rank == 0:
        molecule.save()
    return molecule


def get_BeH2_object(r=1.3038, multiplicity = 1, charge = 0, basis_type = "sto-3g"):
    geometry = [ ['Be', [0,  0, 0]], ['H', [r,  0, 0]], ['H', [-r,  0, 0]]] 
    molecule_type = "BeH2"

    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    filename = directory + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)

    molecule = MolecularData(geometry, basis_type, multiplicity , charge, 
                                filename=filename,)
    molecule.save()
    return molecule    

import numpy as np
def get_NH3_object(r=1.06995934,
                    multiplicity = 1, 
                    charge = 0, 
                    basis_type = "sto-3g"
                    ):
    # actual distance N-H is 1.06995934
    vN=np.array([0,  0, 0.149]) * (r/1.07)
    vH1=np.array([0,  0.947, -0.349]) * (r/1.07)
    vH2=np.array([0.820, -0.474, -0.349]) * (r/1.07)
    vH3=np.array([-0.820, -0.474, -0.349]) * (r/1.07)
    geometry = [ ['N', vN.tolist()], ['H', vH1.tolist()], ['H', vH2.tolist()], ['H', vH3.tolist()]] 

    molecule_type = "NH3"

    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    filename = directory + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)

    molecule = MolecularData(geometry, basis = basis_type,  multiplicity = multiplicity, filename = filename)
    molecule.save()
    return molecule    

def get_N2_object(r, multiplicity = 1, charge = 0, basis_type = "sto-3g", rank = 0):
    geometry = [ ['N', [0,  0, 0]], ['N', [r,  0, 0]]] 
    molecule_type = "N2"
    
    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    filename = directory + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)

    molecule = MolecularData(geometry, basis_type, multiplicity , charge, 
                                filename=filename,)
    if rank == 0:                              
        molecule.save()
    return molecule

def get_C2_object(r=1.2691, multiplicity = 1, charge = 0, basis_type = "sto-3g"):
    geometry = [ ['C', [0,  0, 0]], ['C', [r,  0, 0]]] 
    molecule_type = "C2"

    directory = "molecules/" + molecule_type + "_" + basis_type + "/"
    assert_directory_exists(directory)    
    
    filename = "molecules/" + molecule_type + "_" + basis_type + "/" + molecule_type + "_" + basis_type + "_" + "r_%.2f"%(r)
    molecule = MolecularData(geometry, basis_type, multiplicity , charge, 
                                filename=filename,)
    molecule.save()
    return molecule