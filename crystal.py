"""
This module contains functions for working with crystal field Hamiltonians
"""
import time
from functools import reduce, lru_cache, partial
from itertools import product
from collections import namedtuple
from fractions import Fraction
import warnings
import h5py

from jax import numpy as jnp
from jax import scipy as jscipy
from jax import grad, jacfwd, jit, vmap
from jax.lax import stop_gradient
from jax.config import config

import numpy as np

from jax import core, ad, lax
import jax.numpy.linalg as la

from sympy.physics.wigner import wigner_3j, wigner_6j
import scipy.special as ssp
from scipy import integrate

from hpc_suite.store import Store

import utils as ut
from basis import unitary_transform, cartesian_op_squared, rotate_cart, \
    sfy, calc_ang_mom_ops, make_angmom_ops_from_mult, project_angm_basis, \
    Term, Level, couple, sf2ws, sf2ws_amfi, extract_blocks, from_blocks, \
    dissect_array, ANGM_SYMBOLS, TOTJ_SYMBOLS

from pylanczos import PyLanczos

N_TOTAL_CFP_BY_RANK = {2: 5, 4: 14, 6: 27}
RANK_BY_N_TOTAL_CFP = {val: key for key, val in N_TOTAL_CFP_BY_RANK.items()}
HARTREE2INVCM = 219474.6

config.update("jax_enable_x64", True)






class FromFile:

    def __init__(self, h_file, **kwargs):

        self.h_file = h_file

        with h5py.File(self.h_file, 'r') as h:
            ops = {op: h[op][...] for op in ['hamiltonian', 'spin', 'angm']}

        super().__init__(ops, **kwargs)


class MagneticSusceptibility(Store):

    def __init__(self, ops, temperatures=None, field=None, differential=False,
                 iso=True, powder=False, chi_T=False, units='cm^3 / mol',
                 fmt='% 20.13e'):

        self.ops = ops
        #print(self.ops['hamiltonian'])
        self.temperatures = temperatures

        # basis options
        self.field = field
        self.differential = differential
        self.iso = iso
        self.powder = powder
        self.chi_T = chi_T

        title = "chi_T" if self.chi_T else "chi"
        description = " ".join(["Temperature-dependent"] +
                               (["differential"] if self.differential else []) +
                               (["isotropic"] if self.iso else []) +
                               ["molecular susceptibility"] +
                               (["tensor"] if not self.iso else []) +
                               (["times temperature"] if self.chi_T else []) +
                               [f"at {field} mT"])

        super().__init__(title, description, label=(), units=units, fmt=fmt)

    def evaluate(self):
        ops = self.ops
        #print(ops['spin'])
        tensor_func = make_susceptibility_tensor(
            hamiltonian=ops['hamiltonian'],
            spin=ops['spin'], angm=ops['angm'],
            field=self.field)

        chi_list = [tensor_func(temp) for temp in self.temperatures]

        data = {}

        if True: # Change from True if you want to have the scalar of chi, false for vector
            for temp, chi_tensor in zip(self.temperatures, chi_list):
                scalar_chi = jnp.trace(chi_tensor) / 3
                data[temp] = scalar_chi
        else:
            data = {temp: chi for temp, chi in zip(self.temperatures, chi_list)}

        return data



    def __iter__(self):
        yield from ((lab, dat) for dat, lab in zip(*self.evaluate(**self.ops)))


class MagneticSusceptibilityFromFile(FromFile, MagneticSusceptibility):
    pass




def magmom(spin, angm):
    muB = 0.5  # atomic units
    g_e = 2.002319
    return muB * (angm + g_e * spin)


def eprg_tensor(spin, angm):
    muB = 0.5  # atomic units
    magm = magmom(spin, angm) / muB
    return 2 * jnp.einsum('kij,lji->kl', magm, magm).real


def zeeman_hamiltonian(spin, angm, field):
    """Compute Zeeman Hamiltonian in atomic units.

    Parameters
    ----------
    spin : np.array
        Spin operator in the SO basis.
    angm : np.array
        Orbital angular momentum operator in the SO basis.
    field : np.array
        Magnetic field in mT.

    Returns
    -------
    np.array
        Zeeman Hamiltonian matrix.
    """

    au2mT = 2.35051756758e5 * 1e3  # mTesla / au

    # calculate zeeman operator and convert field in mT to T
    #print("ZEEMAN FOR CRYSTAL.PY")
    z = jnp.einsum('i,imn->mn', jnp.array(field) / au2mT, magmom(spin, angm))
    #print(z)
    return z


def Gtensor(spin, angm):
    muB = 0.5  # atomic units
    magn = magmom(spin, angm)
    return 2 / muB * jnp.einsum('kuv,lvu', magn, magn)


# @partial(jit, static_argnames=['differential', 'algorithm'])
def susceptibility_tensor(temp, hamiltonian, spin, angm, field=0.,
                          differential=True, algorithm=None):
    """Differential molar magnetic susceptipility tensor under applied magnetic
    field along z, or conventional susceptibility tensor where each column
    represents the magnetic response under applied magnetic field along x, y or
    z.

    Parameters
    ----------
    temp : float
        Temperature in Kelvin.
    hamiltonian : np.array
        Electronic Hamiltonian in atomic units.
    spin : np.array
        Spin operator in the SO basis.
    angm : np.array
        Orbital angular momentum operator in the SO basis.
    field : float
        Magnetic field in mT at which susceptibility is measured.
    differential : bool
        If True, calculate differential susceptibility.
    algorithm : {'eigh', 'expm'}
        Algorithm for the computation of the partition function.

    Returns
    -------
    3x3 np.array

    """
    a0 = 5.29177210903e-11  # Bohr radius in m
    c0 = 137.036  # a.u.
    mu0 = 4 * np.pi / c0**2  # vacuum permeability in a.u.
    au2mT = 2.35051756758e5 * 1e3  # mTesla / au

    # [hartree] / [mT mol] * [a.u.(velocity)^2] / [mT]
    algorithm = algorithm or ('expm' if differential else 'eigh')
    mol_mag = partial(molecular_magnetisation, temp, hamiltonian, spin, angm,
                      algorithm=algorithm)

    if differential:
        chi = mu0 * jacfwd(mol_mag)(jnp.array([0., 0., field]))
    else:
        # conventional susceptibility at finite field
        chi = mu0 * jnp.column_stack([mol_mag(fld) / field
                                      for fld in field * jnp.identity(3)])

    # [cm^3] / [mol] + 4*pi for conversion from SI cm3
    return (a0 * 100)**3 * au2mT**2 * chi / (4 * np.pi)


def make_susceptibility_tensor(hamiltonian, spin, angm, field=0.):
    """Differential molar magnetic susceptipility tensor under applied magnetic
    field along z, or conventional susceptibility tensor where each column
    represents the magnetic response under applied magnetic field along x, y or
    z. Maker function for partial evaluation of matrix eigen decomposition.


    Parameters
    ----------
    hamiltonian : np.array
        Electronic Hamiltonian in atomic units.
    spin : np.array
        Spin operator in the SO basis.
    angm : np.array
        Orbital angular momentum operator in the SO basis.
    field : float
        Magnetic field in mT at which susceptibility is measured.

    Returns
    -------
    3x3 np.array

    """
    a0 = 5.29177210903e-11  # Bohr radius in m
    c0 = 137.036  # a.u.
    mu0 = 4 * np.pi / c0**2  # vacuum permeability in a.u.
    au2mT = 2.35051756758e5 * 1e3  # mTesla / au

    # [hartree] / [mT mol] * [a.u.(velocity)^2] / [mT]

    mol_mag = [make_molecular_magnetisation(hamiltonian, spin, angm, fld)
               for fld in field * jnp.identity(3)]

    # conventional susceptibility at finite field
    def susceptibility_tensor(temp):
        chi = mu0 * jnp.column_stack([mol_mag[comp](temp) / field for comp in range(3)])
        # [cm^3] / [mol] + 4*pi for conversion from SI cm3
        return (a0 * 100)**3 * au2mT**2 * chi / (4 * np.pi)

    return susceptibility_tensor



def make_molecular_magnetisation(hamiltonian, spin, angm, field):
    """ Molar molecular magnetisation in [hartree] / [mT mol] maker function
    for partial evaluation of matrix eigen decomposition.

    Parameters
    ----------
    hamiltonian : np.array
        Electronic Hamiltonian in atomic units.
    spin : np.array
        Spin operator in the SO basis.
    angm : np.array
        Orbital angular momentum operator in the SO basis.
    field : np.array
        Magnetic field in mT at which susceptibility is measured. If None,
        returns differential susceptibility.
    """
    estimate = 16 #number of eigenvalues returned by the approximations
    Na = 6.02214076e23  # 1 / mol
    kb = 3.166811563e-6  # hartree / K
    au2mT = 2.35051756758e5 * 1e3  # mTesla / au
    #print("EINZELKIND")
    h_total = hamiltonian + zeeman_hamiltonian(spin, angm, field)
    #h_total = [[4.0, 2.0],
    #          [2.0, 5.0]]
    #print("Full Ham")
    
    #print(h_total)
    # condition matrix by diagonal shift
    start_numpy = time.time()
    eig, vec = jnp.linalg.eigh(np.array(h_total))
    end_numpy = time.time()
    numpy_time = end_numpy - start_numpy

    print("NumPy took", numpy_time, "seconds")

    start_davidson = time.time()
    eig1, vec1 = davidson(h_total, estimate)
    end_davidson = time.time()
    davidson_time = end_davidson - start_davidson

    print("Davidson took", davidson_time, "seconds")
    start_l = time.time()
    #eig2, vec2 = lanczos(h_total,estimate)
    #print("Lanczos took",time.time()-start_l )
    if numpy_time < davidson_time:
        print("NumPy is quicker.")
    elif davidson_time < numpy_time:
        print("Davidson is quicker.")
    else:
        print("Both NumPy and Davidson took the same amount of time.")


    if len(eig) == len(eig1):
        print("Eigenvalues are the same = "+str(np.allclose(eig,eig1)))
        print("Eigenvectors are the same = "+str(np.allclose(vec,vec1)))
    def molecular_magnetisation(temp):
        beta = 1 / (kb * temp)  # hartree
        eig_shft = eig - stop_gradient(eig[0])
        expH = vec @ jnp.diag(jnp.exp(-beta * eig_shft)) @ vec.T.conj()
        Z = jnp.sum(jnp.exp(-beta * eig_shft))
        dZ = -jnp.einsum('ij,mji', expH, magmom(spin, angm) / au2mT).real
        return Na * dZ / Z

    return molecular_magnetisation


def lanczos(matrix,approxEigs):
    '''
    Method to calculate Lanczos diagonalisation
    Method 6
    Returns eigenvalues and eigenvectors
    '''

    start_lanc = time.time() # Start timer
    #matrix = self.getHam()
    engine = PyLanczos(np.array(matrix), True, approxEigs)  # Find maximum eigenpairs
    eigenvalues, eigenvectors = engine.run() #These eigenvectors are wrong/different to numpy's
    finish_lanc = time.time() # End timer
    #print("lanczos = ",finish_lanc - start_lanc, "seconds")
    eigenvalues = map(np.real,eigenvalues)
    eigenvalues = list(eigenvalues)

    return np.sort(eigenvalues[:approxEigs]), np.sort(eigenvectors[:approxEigs])


def davidson(A, neig):
    n = A.shape[0]
    tol = 1e-9             # Convergence tolerance
    mmax = 18              # Maximum number of iterations

    # Setup the subspace trial vectors
    k = neig

    t = np.eye(n, k) # initial trial vectors
    v = np.zeros((n, n)) # holder for trial vectors as iterations progress
    I = np.eye(n) # n*n identity matrix
    ritz = np.zeros((n, n))
    f = np.zeros((n, n))
    
    # Lists to store eigenvalues and eigenvectors
    eigenvalues = []
    eigenvectors = []

    #-------------------------------------------------------------------------------
    # Begin iterations  
    #-------------------------------------------------------------------------------
    iter = 0
    for m in range(k, mmax, k):
        iter = iter + 1
        if iter == 1:  # for the first iteration, add normalized guess vectors to matrix v
            for l in range(m):
                v[:, l] = t[:, l] / (np.linalg.norm(t[:, l]))
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        T = np.linalg.multi_dot([v[:, :m].T, A, v[:, :m]]) # selects the fastest evaluation order
        w, vects = np.linalg.eigh(T) # Diagonalize the subspace Hamiltonian
        jj = 0
        s = w.argsort()
        ss = w[s]
        #***************************************************************************
        # For each eigenvector of T, build a Ritz vector, precondition it and check
        # if the norm is greater than a set threshold.
        #***************************************************************************
        for ii in range(m): # for each new eigenvector of T
            f = np.diag(1. / np.diag((np.diag(np.diag(A)) - w[ii] * I)))
            ritz[:, ii] = np.dot(f, np.linalg.multi_dot([(A - w[ii] * I), v[:, :m], vects[:, ii]]))
            if np.linalg.norm(ritz[:, ii]) > 1e-7:
                ritz[:, ii] = ritz[:, ii] / (np.linalg.norm(ritz[:, ii]))
                v[:, m + jj] = ritz[:, ii]
                jj = jj + 1
        q, r = np.linalg.qr(v[:, :m + jj - 1])
        for kk in range(m + jj - 1):
            v[:, kk] = q[:, kk]

        if iter == 1: 
            check_old = ss[:neig]
            check_new = 1
        elif iter == 2:
            check_new = ss[:neig]
        else: 
            check_old = check_new
            check_new = ss[:neig]
        check = np.linalg.norm(check_new - check_old)
        if check < tol:       
            break

    # Store eigenvalues and eigenvectors
    for ii in range(neig):
        eigenvalues.append(ss[ii])
        eigenvectors.append(vects[:, ii])

    end = time.time()
    
    start = time.time()
    eig, eigvecs = np.linalg.eig(A)
    end = time.time() 
    s = eig.argsort()
    ss = eig[s]
    # Return eigenvalues and eigenvectors
    return eigenvalues, eigenvectors



fileName = "ops(1).hdf5"
temperatures1 = [1.1]
field = 0.8


angmomSus = MagneticSusceptibilityFromFile(fileName,temperatures=temperatures1,field=0.8 , differential = True)

print(angmomSus.evaluate())