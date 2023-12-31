from openfermion.ops import InteractionOperator
import numpy
#import numpy as np
import copy
from openfermion import get_fermion_operator

from pyscf import gto
try:
    # this worked for openfermion v0.10.0, not for v1.1.0
    from openfermion.hamiltonians import MolecularData
except:
    # this worked for openfermion v1.1.0
    from openfermion import MolecularData
from pyscf import ao2mo

def get_molecular_hamiltonian_generalAS(molecule, 
                                        occupied_spinorbitals = None, 
                                        active_spinorbitals = None,
                                        return_coefficients = False,
                                        verbose = 0,
                                        pyscf_mf = None,
                                        debug_spin_nonequiv = False,
                                        ):


    if isinstance(molecule, MolecularData):
        one_body_integrals, two_body_integrals = molecule.get_integrals()
        nalpha, nbeta = molecule.get_n_alpha_electrons(), molecule.get_n_beta_electrons()
        en_nuc = molecule.nuclear_repulsion

    elif isinstance(molecule, gto.Mole):
        if pyscf_mf is not None:
            one_body_integrals, two_body_integrals = compute_integrals(molecule, pyscf_mf)
            en_nuc = float(molecule.energy_nuc())
        else:

            from pyscf import scf
            if molecule.spin == 0:
                mf = scf.RHF(molecule)
            else:
                mf = scf.ROHF(molecule)
            mf.run(verbose = 0)

            en_nuc = molecule.energy_nuc()
            one_body_integrals, two_body_integrals = compute_integrals(molecule, mf)
        nalpha, nbeta = molecule.nelec

    if occupied_spinorbitals is None and active_spinorbitals is None:
        constant = en_nuc
        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals, two_body_integrals)    

    else:
        #if nalpha == nbeta:
        core_adjustment, one_body_integrals_alpha, one_body_integrals_beta, two_body_integrals_global = \
            get_active_space_integrals_GAS(
                one_body_integrals, 
                two_body_integrals, 
                occupied_spinorbitals, 
                active_spinorbitals, 
                verbose, 
                debug_spin_nonequiv
                )
        constant = en_nuc + core_adjustment

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial_GAS(
            one_body_integrals_alpha, 
            one_body_integrals_beta, 
            two_body_integrals_global, 
            occupied_spinorbitals, 
            active_spinorbitals)
        """
        else:
            constant_tmp = en_nuc
            one_body_coefficients_tmp, two_body_coefficients_tmp = spinorb_from_spatial(
                one_body_integrals, two_body_integrals)  

            constant, one_body_coefficients, two_body_coefficients = general_active_space_coefficients(
                constant_tmp, 
                one_body_coefficients_tmp, 
                two_body_coefficients_tmp,
                occupied_spinorbitals,
                active_spinorbitals
            )
        """                            
            
    

    if return_coefficients:
        return constant, one_body_coefficients, 1/2 * two_body_coefficients
    else:   
        return InteractionOperator(
        constant, one_body_coefficients, 1 / 2 * two_body_coefficients)


def general_active_space_coefficients(
    constant, 
    one_body_tensor, 
    two_body_tensor, 
    occupied_indices, 
    active_indices
    ):
    constant_new = copy.deepcopy(constant)
    one_body_new = numpy.copy(one_body_tensor)    
    two_body_new = numpy.copy(two_body_tensor)    
    
    for i in occupied_indices:
        constant_new += one_body_tensor[i, i]
        for j in occupied_indices:
            constant_new += (two_body_tensor[i, j, j, i] - two_body_tensor[i, j, i, j])/2

    # update one body coefficients
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                one_body_new[u, v] += (two_body_tensor[i, u, v, i] - 
                                                                   two_body_tensor[i, u, i, v])
                #one_body_new[u, v] += (two_body_tensor[i, u, v, i] + two_body_tensor[u, i, i, v] - two_body_tensor[i, u, i, v]-  two_body_tensor[u, i, v, i])/2
                    

    one_body_new = one_body_new[numpy.ix_(active_indices, active_indices)]
    two_body_new = two_body_new[numpy.ix_(active_indices, active_indices, active_indices, active_indices)]                
    
    return constant_new, one_body_new, two_body_new
    #int_op = InteractionOperator(constant_new, one_body_new, 0.5 * two_body_new)
    #return get_fermion_operator(int_op)

def get_active_space_integrals_GAS(one_body_integrals,
                               two_body_integrals,
                               occupied_spinorbitals=None,
                               active_spinorbitals=None,
                               verbose = 0,
                               debug_spin_nonequiv = False
                               ):
                               #occupied_indices=None,
                               #active_indices=None):
    
    """Restricts a molecule at a spatial orbital level to an active space
    This active space may be defined by a list of active indices and
        doubly occupied indices. Note that one_body_integrals and
        two_body_integrals must be defined
        n an orthonormal basis set.
    Args:
        one_body_integrals: One-body integrals of the target Hamiltonian
        two_body_integrals: Two-body integrals of the target Hamiltonian
        occupied_indices: A list of spatial orbital indices
            indicating which orbitals should be considered doubly occupied.
        active_indices: A list of spatial orbital indices indicating
            which orbitals should be considered active.
    Returns:
        tuple: Tuple with the following entries:
        **core_constant**: Adjustment to constant shift in Hamiltonian
        from integrating out core orbitals
        **one_body_integrals_new**: one-electron integrals over active
        space.
        **two_body_integrals_new**: two-electron integrals over active
        space.
    """
    # Fix data type for a few edge cases
    #occupied_indices = [] if occupied_indices is None else occupied_indices
    #if (len(active_indices) < 1):
        #raise ValueError('Some active indices required for reduction.')    
    occupied_spinorbitals = [] if occupied_spinorbitals is None else occupied_spinorbitals
    if (len(active_spinorbitals) < 1):
        raise ValueError('Some active indices required for reduction.')
    
    # convert spinorbitals (<=n_qubit) into indices (<= n_qubit/2)
    # e.g. [0, 1, 2, 3] -> [0, 1], [0,1]
    occupied_indices_alpha = [i//2 for i in occupied_spinorbitals if i%2 ==0]
    occupied_indices_beta = [i//2 for i in occupied_spinorbitals if i%2 ==1]
    active_indices_alpha = [i//2 for i in active_spinorbitals if i%2 ==0]
    active_indices_beta = [i//2 for i in active_spinorbitals if i%2 ==1]

    if verbose == 1:
        print("\noccupied_indices_alpha = ", occupied_indices_alpha)
        print("occupied_indices_beta = ", occupied_indices_beta)
        print("active_indices_alpha = ", active_indices_alpha)
        print("active_indices_beta = ", active_indices_beta)

    alpha_beta_equivalent = (occupied_indices_alpha == occupied_indices_beta and \
                             active_indices_alpha == active_indices_beta)
    
    if alpha_beta_equivalent and not debug_spin_nonequiv:
        occupied_indices = occupied_indices_alpha
        active_indices = active_indices_alpha
        
        # Determine core constant
        core_constant = 0.0
        for i in occupied_indices:
            core_constant += 2 * one_body_integrals[i, i]
            for j in occupied_indices:
                core_constant += (2 * two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])
                
        # Modified one electron integrals
        one_body_integrals_new = numpy.copy(one_body_integrals)
        for u in active_indices:
            for v in active_indices:
                for i in occupied_indices:
                    one_body_integrals_new[u, v] += (
                        2 * two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])
                    
        one_body_integrals_alpha_new = numpy.copy(one_body_integrals_new)
        one_body_integrals_beta_new = numpy.copy(one_body_integrals_new)
        
    else:
        #raise Exception("This fuction is not correct when n_alpha != n_beta. Use `construct_active_space` instead.")
        # Determine core constant
        core_constant = 0.0
        for i in occupied_indices_alpha:
            core_constant += one_body_integrals[i, i]

            # same spin
            for j in occupied_indices_alpha:
                core_constant += (two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])/2
            # mixed
            for j in occupied_indices_beta:
                core_constant += (two_body_integrals[i, j, j, i])/2
                # updated on 2021/11/13
                #core_constant += (two_body_integrals[i, j, j, i] -
                                  #two_body_integrals[i, j, i, j])/2


        for i in occupied_indices_beta:
            core_constant += one_body_integrals[i, i]

            # same spin
            for j in occupied_indices_beta:
                core_constant += (two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])/2
            # mixed
            for j in occupied_indices_alpha:
                core_constant += (two_body_integrals[i, j, j, i])/2
                # updated on 2021/11/15
                #core_constant += (two_body_integrals[i, j, j, i] -
                                  #two_body_integrals[i, j, i, j])/2


        # Modified one electron integrals
        one_body_integrals_alpha_new = numpy.copy(one_body_integrals)
        for u in active_indices_alpha:
            for v in active_indices_alpha:
                # same spin
                for i in occupied_indices_alpha:
                    one_body_integrals_alpha_new[u, v] += (
                        two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])
                # mix spin
                for i in occupied_indices_beta:
                    one_body_integrals_alpha_new[u, v] += (
                        two_body_integrals[i, u, v, i])                    
                    # updated on 2021/11/15
                    #one_body_integrals_alpha_new[u, v] += (
                        #two_body_integrals[i, u, v, i] -
                        #two_body_integrals[i, u, i, v])



        one_body_integrals_beta_new = numpy.copy(one_body_integrals)
        for u in active_indices_beta:
            for v in active_indices_beta:

                # same spin
                for i in occupied_indices_beta:
                    one_body_integrals_beta_new[u, v] += (
                        two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])
                # mix spin
                for i in occupied_indices_alpha:
                    one_body_integrals_beta_new[u, v] += (
                        two_body_integrals[i, u, v, i])
                    # updated on 2021/11/13
                    #one_body_integrals_beta_new[u, v] += (
                        #two_body_integrals[i, u, v, i] -
                        #two_body_integrals[i, u, i, v])


    # Restrict integral ranges and change M appropriately
    return (core_constant,
            one_body_integrals_alpha_new[numpy.ix_(active_indices_alpha, active_indices_alpha)],
            one_body_integrals_beta_new[numpy.ix_(active_indices_beta, active_indices_beta)],
            two_body_integrals)    

from openfermion.config import EQ_TOLERANCE
def spinorb_from_spatial_GAS(one_body_integrals_alpha, one_body_integrals_beta, two_body_integrals, occupied_spinorbitals=None, active_spinorbitals=None):
    #n_qubits = 2 * one_body_integrals.shape[0]
    n_alpha = one_body_integrals_alpha.shape[0]
    n_beta = one_body_integrals_beta.shape[0]
    n_qubits = n_alpha + n_beta
    
    n_qubits_fci = 2 * two_body_integrals.shape[0]
    
    active_spinorbitals_alpha = [i for i in active_spinorbitals if i%2 ==0]
    active_spinorbitals_beta = [i for i in active_spinorbitals if i%2 ==1]

    # Initialize Hamiltonian coefficients.
    # 2-body coefficient extracted later.
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
    two_body_coefficients = numpy.zeros(
        (n_qubits_fci, n_qubits_fci, n_qubits_fci, n_qubits_fci))
    
    # 1-body for alpha spin
    for pp in range(n_alpha):
        p = active_spinorbitals.index(active_spinorbitals_alpha[pp])
        for qq in range(n_alpha):
            q = active_spinorbitals.index(active_spinorbitals_alpha[qq])
            one_body_coefficients[p, q] = one_body_integrals_alpha[pp, qq]
                    
    # 1-body for beta spin
    for pp in range(n_beta):
        p = active_spinorbitals.index(active_spinorbitals_beta[pp])
        for qq in range(n_beta):
            q = active_spinorbitals.index(active_spinorbitals_beta[qq])
            one_body_coefficients[p, q] = one_body_integrals_beta[pp, qq]
            
    # for 2-body coefficients, we simply calculate all the term and extract them later
    # Loop through integrals: alpha spin
    for p in range(n_qubits_fci // 2):
        for q in range(n_qubits_fci // 2):
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits_fci // 2):
                for s in range(n_qubits_fci // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

    two_body_coefficients = two_body_coefficients[numpy.ix_(active_spinorbitals, active_spinorbitals, active_spinorbitals, active_spinorbitals, )]

    # Truncate.
    one_body_coefficients[
        numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    return one_body_coefficients, two_body_coefficients            



def spinorb_from_spatial(one_body_integrals, two_body_integrals):
    n_qubits = 2 * one_body_integrals.shape[0]

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = numpy.zeros((n_qubits, n_qubits))
    two_body_coefficients = numpy.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s])
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s])

    # Truncate.
    one_body_coefficients[
        numpy.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        numpy.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    return one_body_coefficients, two_body_coefficients    

############################################################
# for raw pyscf objects
############################################################
import numpy
from functools import reduce
def compute_integrals(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.
    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.
    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(numpy.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = numpy.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals    