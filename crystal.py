"""
This module contains functions for working with crystal field Hamiltonians
"""

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


N_TOTAL_CFP_BY_RANK = {2: 5, 4: 14, 6: 27}
RANK_BY_N_TOTAL_CFP = {val: key for key, val in N_TOTAL_CFP_BY_RANK.items()}
HARTREE2INVCM = 219474.6

config.update("jax_enable_x64", True)


@lru_cache(maxsize=None)
def recursive_a(k, q, m):
    """
    Given k,q,m this function
    calculates and returns the a(k,q-1,m)th
    Ryabov coefficient by recursion

    Parameters
    ----------
    k : int
        k value (rank)
    q : int
        q value (order)
    m : int
        m value

    Returns
    -------
    np.ndarray
        a(k,q,m) values for each power of X=J(J+1) (Ryabov) up to k+1
    """

    coeff = np.zeros(k+1)

    # Catch exceptions/outliers and end recursion
    if k == q-1 and m == 0:
        coeff[0] = 1
    elif q-1 + m > k:
        pass
    elif m < 0:
        pass
    else:
        # First and second terms
        coeff += (2*q+m-1)*recursive_a(k, q+1, m-1)
        coeff += (q*(q-1) - m*(m+1)/2) * recursive_a(k, q+1, m)

        # Third term (summation)
        for n in range(1, k-q-m+1):
            # First term in sum of third term
            coeff[1:] += (-1)**n * (
                            ut.binomial(m+n, m) * recursive_a(k, q+1, m+n)[:-1]
                        )
            # Second and third term in sum
            coeff += (-1)**n * (
                        - ut.binomial(m+n, m-1) - ut.binomial(m+n, m-2)
                    ) * recursive_a(k, q+1, m+n)

    return coeff


def get_ryabov_a_coeffs(k_max):

    """
    Given k_max this function calculates all possible values
    of a(k,q,m) for each power (i) of X=J(J+1)

    Parameters
    ----------
    k_max : int
        maximum k (rank) value

    Returns
    -------
    np.ndarray
        All a(k,q,m,i)
    np.ndarray
        Greatest common factor of each a(k,q,:,:)
    """

    a = np.zeros([k_max, k_max+1, k_max+1, k_max+1])
    f = np.zeros([k_max, k_max+1])

    # Calculate all a coefficients
    for k in range(1, k_max + 1):
        for qit, q in enumerate(range(k, -1, -1)):
            for m in range(k-q + 1):
                a[k-1, qit, m, :k+1] += recursive_a(k, q+1, m)

    # Calculate greatest common factor F for each a(k,q) value
    for k in range(1, k_max + 1):
        for qit, q in enumerate(range(k, -1, -1)):
            allvals = a[k-1, qit, :, :].flatten()
            nzind = np.nonzero(allvals)
            if np.size(nzind) > 0:
                f[k-1, qit] = reduce(ut.GCD, allvals[nzind])

    return a, f


def calc_stev_ops(k_max, J, jp, jm, jz):
    """
    Calculates all Stevens operators Okq with k even and odd from k=1 to k_max
    k_max must be <= 12 (higher rank parameters require quad precision floats)

    Parameters
    ----------
    k_max : int
        maximum k value (rank)
    J : int
        J quantum number
    jp : np.array
        Matrix representation of angular momentum operator
    jm : np.array
        Matrix representation of angular momentum operator
    jz : np.array
        Matrix representation of angular momentum operator

    Returns
    -------
    np.ndarray
        Stevens operators shape = (n_k, n_q, (2J+1), (2J+1))
            ordered k=1 q=-k->k, k=2 q=-k->k ...
    """

    # Only k <= 12 possible at double precision
    k_max = min(k_max, 12)

    # Get a(k,q,m,i) coefficients and greatest common factors
    a, f = get_ryabov_a_coeffs(k_max)

    # Sum a(k,q,m,i) coefficients over powers of J to give a(k,q,m)
    a_summed = np.zeros([k_max, k_max+1, k_max+1])

    for i in range(0, k_max+1):
        a_summed += a[:, :, :, i] * float(J*(J+1))**i

    _jp = np.complex128(jp)
    _jm = np.complex128(jm)
    _jz = np.complex128(jz)

    n_states = int(2*J+1)

    okq = np.zeros([k_max, 2*k_max+1, n_states, n_states], dtype=np.complex128)

    # Calulate q operators both + and - at the same time
    for kit, k in enumerate(range(1, k_max + 1)):
        # New indices for q ordering in final okq array
        qposit = 2*k + 1
        qnegit = -1
        for qit, q in enumerate(range(k, -1, -1)):
            qposit -= 1
            qnegit += 1
            if k % 2:  # Odd k, either odd/even q
                alpha = 1.
            elif q % 2:  # Even k, odd q
                alpha = 0.5
            else:  # Even k, even q
                alpha = 1.

            # Positive q
            for m in range(k-q + 1):
                okq[kit, qposit, :, :] += a_summed[kit, qit, m]*(
                    (
                        la.matrix_power(_jp, q)
                        + (-1.)**(k-q-m)*la.matrix_power(_jm, q)
                    ) @ la.matrix_power(_jz, m)
                )

            okq[kit, qposit, :, :] *= alpha/(2*f[kit, qit])

            # Negative q
            if q != 0:
                for m in range(k-q + 1):
                    okq[kit, qnegit, :, :] += a_summed[kit, qit, m]*(
                        (
                            la.matrix_power(_jp, q)
                            - (-1.)**(k-q-m)*la.matrix_power(_jm, q)
                        ) @ la.matrix_power(_jz, m)
                    )

                okq[kit, qnegit, :, :] *= alpha/(2j*f[kit, qit])

    return okq


def calc_stev_ops_jax(*args):
    return calc_stev_ops_jax_p.bind(*args)

def calc_stev_ops_jax_jvp(primals, tangents):
    return lax.zeros_like_array(primals)

calc_stev_ops_jax_p = core.Primitive("calc_stev_ops_jax")
calc_stev_ops_jax_p.def_impl(calc_stev_ops)
ad.defjvp(calc_stev_ops_jax_p, calc_stev_ops_jax_jvp)


def load_CFPs(f_name, style="phi", k_parity="even"):
    """
    Loads Crystal Field Parameters (CFPs) from file

    Parameters
    ----------
    f_name : str
        file name to load CFPs from
    style : str {'phi','raw'}
        Style of CFP file:
            Phi = Chilton's PHI Program input file
            raw = list of CFPs arranged starting with smallest value of k
                  following the scheme k=k_min q=-k->k, k=k_min+1 q=-k->k ...
    k_parity : str {'even', 'odd', 'both'}
        Indicates type of k values
            e.g. k=2,4,6,... or k=1,3,5,... or k=1,2,3...

    Returns
    -------
    np.ndarray
        CFPs with shape = (n_k, n_q)
            ordered k=k_min q=-k->k, k=k_min+mod q=-k->k ...
            where mod is 1 or 2 depending upon k_parity
    """

    _CFPs = []
    if style == "phi":
        # PHI does not support odd rank cfps
        k_parity = "even"
        # Read in CFPs, and k and q values
        kq = []
        # site, k, q, Bkq
        with open(f_name, 'r') as f:
            for line in f:
                if '****crystal' in line.lower():
                    line = next(f)
                    while "****" not in line:
                        kq.append(line.split()[1:3])
                        _CFPs.append(line.split()[3])
                        line = next(f)
                    break
        kq = [[int(k), int(q)] for [k, q] in kq]
        _CFPs = np.array([float(CFP) for CFP in _CFPs])

        # Include zero entries for missing CFPs
        # and reorder since PHI files might be in wrong order

        # find largest k and use to set size of array
        k_max = np.max(kq[0])
        n_cfps = np.sum([2*k + 1 for k in range(k_max, 0, -1)])
        CFPs = np.zeros([n_cfps])
        if k_parity == "even":
            for CFP, [k, q] in zip(_CFPs, kq):
                CFPs[_even_kq_to_num(k, q)] = CFP
        elif k_parity == "odd":
            for CFP, [k, q] in zip(_CFPs, kq):
                CFPs[_odd_kq_to_num(k, q)] = CFP
        else:
            for CFP, [k, q] in zip(_CFPs, kq):
                CFPs[_kq_to_num(k, q)] = CFP

    elif style == "raw":
        CFPs = np.loadtxt(f_name)

    return CFPs


def calc_HCF(J, cfps, stev_ops, k_max=False, oef=[]):
    """
    Calculates and diagonalises crystal field Hamiltonian (HCF)
    using CFPs Bkq and Stevens operators Okq, where k even and ranges 2 -> 2j

    Hamiltonian is sum_k (sum_q (oef_k*Bkq*Okq))

    Parameters
    ----------
    J : float
        J quantum number
    cfps : np.array
        Even k crystal Field parameters, size = (n_k*n_q)
        ordered k=2 q=-k->k, k=4 q=-k->k ...
    np.ndarray
        Stevens operators, shape = (n_k, n_q, (2J+1), (2J+1))
        ordered k=2 q=-k->k, k=4 q=-k->k ...
    k_max : int, default = 2*J
        Maximum value of k to use in summation
    oef : np.ndarray, optional
        Operator equivalent factors for each CFP i.e. 27 CFPs = 27 OEFs
        size = (n_k*n_q), ordered k=2 q=-k->k, k=4 q=-k->k ...

    Returns
    -------
    np.array
        Matrix representation of Crystal Field Hamiltonian (HCF)
    np.array
        Eigenvalues of HCF (lowest eigenvalue is zero)
    np.array
        Eigenvectors of HCF
    """

    if not k_max:
        k_max = int(2*J)
        k_max -= k_max % 2
        k_max = min(k_max, 12)

    if not len(oef):
        oef = np.ones(cfps.size)

    # calculate number of states
    n_states = int(2 * J + 1)

    # Form Hamiltonian
    HCF = np.zeros([n_states, n_states], dtype=np.complex128)
    for kit, k in enumerate(range(2, k_max+1, 2)):
        for qit, q in enumerate(range(-k, k+1)):
            HCF += stev_ops[kit, qit, :, :] * cfps[_even_kq_to_num(k, q)] \
                    * oef[_even_kq_to_num(k, q)]

    # Diagonalise
    CF_val, CF_vec = la.eigh(HCF)

    # Set ground energy to zero
    CF_val -= CF_val[0]

    return HCF, CF_val, CF_vec


def calc_oef(n, J, L, S):
    """
    Calculate operator equivalent factors for Stevens Crystal Field
    Hamiltonian in |J, mJ> basis

    Using the approach of
    https://arxiv.org/pdf/0803.4358.pdf

    Parameters
    ----------
    n : int
        number of electrons in f shell
    J : float
        J Quantum number
    L : int
        L Quantum number
    S : float
        S Quantum number

    Returns
    -------
    np.ndarray
        operator equivalent factors for each parameter, size = (n_k*n_q)
        ordered k=2 q=-k->k, k=4 q=-k->k ...
    """

    def _oef_lambda(p, J, L, S):
        lam = (-1)**(J+L+S+p)*(2*J+1)
        lam *= wigner_6j(J, J, p, L, L, S)/wigner_3j(p, L, L, 0, L, -L)
        return lam

    def _oef_k(p, k, n):
        K = 7. * wigner_3j(p, 3, 3, 0, 0, 0)
        if n <= 7:
            n_max = n
        else:
            n_max = n-7
            if k == 0:
                K -= np.sqrt(7)

        Kay = 0
        for j in range(1, n_max+1):
            Kay += (-1.)**j * wigner_3j(k, 3, 3, 0, 4-j, j-4)

        return K*Kay

    def _oef_RedJ(J, p):
        return 1./(2.**p) * (ssp.factorial(2*J+p+1)/ssp.factorial(2*J-p))**0.5

    # Calculate OEFs and store in array
    # Each parameter Bkq has its own parameter
    oef = np.zeros(27)
    k_max = np.min([6, int(2*J)])
    shift = 0
    for k in range(2, k_max+2, 2):
        oef[shift:shift + 2*k+1] = float(_oef_lambda(k, J, L, S))
        oef[shift:shift + 2*k+1] *= float(_oef_k(k, k, n) / _oef_RedJ(J, k))
        shift += 2*k + 1

    return oef


def calc_order_strength(params: list[float]) -> list[float]:
    """
    Calculates per-order strength parameter S_q for a set of Stevens parameters
    up to rank 6
    """

    max_rank = get_max_rank(params)

    # Convert Stevens parameters to Wybourne scheme
    wparams = abs(stevens_to_wybourne(params, max_rank))

    square_params = wparams ** 2

    # Calculate strength within order (S_q)

    sq = np.zeros(len(params))

    # Rank 2 contributions
    sq[0] = 1./5. * square_params[2]
    sq[1] = 1./5. * square_params[3]
    sq[2] = 1./5. * square_params[4]

    # Rank 4 contributions
    if max_rank > 2:
        sq[0] += 1./9. * square_params[9]
        sq[1] += 1./9. * square_params[10]
        sq[2] += 1./9. * square_params[11]
        sq[3] += 1./9. * square_params[12]
        sq[4] += 1./9. * square_params[13]

    # Rank 6 contributions
    if max_rank > 4:
        sq[6] += 1./13. * square_params[26]
        sq[5] += 1./13. * square_params[25]
        sq[4] += 1./13. * square_params[24]
        sq[3] += 1./13. * square_params[23]
        sq[2] += 1./13. * square_params[22]
        sq[1] += 1./13. * square_params[21]
        sq[0] += 1./13. * square_params[20]

    sq = np.sqrt(sq)

    return sq


def calc_rank_strength(params: list[float]) -> list[float]:
    """
    Calculates per-rank strength parameter S^k for a set of Stevens parameters
    up to rank 6
    """

    max_rank = get_max_rank(params)

    # Convert Stevens parameters to Wybourne scheme
    wparams = abs(stevens_to_wybourne(params, max_rank))

    # Calculate strength within rank (S^k)
    sk2 = np.sqrt(np.sum(wparams[:5]**2) / 5.)
    sk4 = np.sqrt(np.sum(wparams[5:14]**2) / 9.)
    sk6 = np.sqrt(np.sum(wparams[14:]**2) / 13.)

    sk = np.array([sk2, sk4, sk6])

    return sk


def calc_total_strength(params: list[float]) -> float:
    """
    Calculates strength parameter S for a set of Stevens parameters up to
    rank 6
    """

    sk = calc_rank_strength(params)

    # Calculate overall strength as weighted sum of S^k values
    S = np.array(np.sqrt(1./3.*(sk[0]**2 + sk[1]**2 + sk[2]**2)))

    return S


def get_max_rank(params):
    """
    Finds maximum rank in a set of parameters, assumes parameters are ordered
    k=2, q=-2...2, k=4, q=-4, ..., 4...
    """

    try:
        max_rank = RANK_BY_N_TOTAL_CFP[len(params)]
    except ValueError:
        raise ValueError("Incorrect number of CFPs")

    return max_rank


def stevens_to_wybourne(CFPs, k_max):
    """
    Transforms Crystal Field parameters from Wybourne notation to
    Stevens notation

    Assumes only even Ranks (k) are present

    Parameters
    ----------
        CFPs : np.ndarray
            CFPs in Stevens notation, shape = (n_k, n_q)
            ordered k=1 q=-k->k, k=2 q=-k->k ...
        k_max : int
            maximum value of k (rank)

    Returns
    -------
        np.ndarray, dtype=complex128
            CFPs in Wybourne notation, shape = (n_k, n_q)
    """

    if k_max > 6:
        raise ValueError("Cannot convert k>6 parameters to Wybourne")

    # Taken from Mulak and Gajek
    lmbda = [
        np.sqrt(6.)/3.,
        -np.sqrt(6.)/6.,
        2.,
        -np.sqrt(6.)/6.,
        np.sqrt(6.)/3.,
        4.*np.sqrt(70.)/35.,
        -2.*np.sqrt(35.)/35.,
        2.*np.sqrt(10.)/5.,
        -2*np.sqrt(5.)/5.,
        8.,
        -2.*np.sqrt(5.)/5.,
        2.*np.sqrt(10.)/5.,
        -2.*np.sqrt(35.)/35.,
        4.*np.sqrt(70.)/35.,
        16.*np.sqrt(231.)/231.,
        -8.*np.sqrt(77.)/231.,
        8.*np.sqrt(14.)/21.,
        -8.*np.sqrt(105.)/105.,
        16.*np.sqrt(105.)/105.,
        -4.*np.sqrt(42.)/21.,
        16.,
        -4.*np.sqrt(42.)/21.,
        16.*np.sqrt(105.)/105.,
        -8.*np.sqrt(105.)/105.,
        8.*np.sqrt(14.)/21.,
        -8.*np.sqrt(77.)/231.,
        16.*np.sqrt(231.)/231.
    ]

    w_CFPs = np.zeros(N_TOTAL_CFP_BY_RANK[k_max], dtype=np.complex128)

    for k in range(2, k_max + 2, 2):
        for q in range(-k, k + 1):
            ind = _even_kq_to_num(k, q)
            neg_ind = _even_kq_to_num(k, -q)
            if q == 0:
                w_CFPs[ind] = lmbda[ind] * CFPs[ind]
            elif q > 0:
                w_CFPs[ind] = lmbda[ind]*(CFPs[ind] + 1j*CFPs[neg_ind])
            elif q < 0:
                w_CFPs[ind] = lmbda[ind]*(-1)**q*(CFPs[neg_ind] - 1j*CFPs[ind])

    return w_CFPs


def _even_kq_to_num(k, q):
    """
    Converts Rank (k) and order (q) to array index
    Assuming that only even ranks are present

    Parameters
    ----------
        k : int
            Rank k
        q : int
            Order q

    Returns
    -------
        int
            Array index
    """

    index = k + q
    for kn in range(1, int(k/2)):
        index += 2*(k-2*kn) + 1

    return index


def _odd_kq_to_num(k, q):
    """
    Converts Rank (k) and order (q) to array index
    Assuming that only odd ranks are present

    Parameters
    ----------
        k : int
            Rank k
        q : int
            Order q

    Returns
    -------
        int
            Array index
    """

    index = 0
    for kn in range(1, k, 2):
        index += 2*kn + 1

    index += q + k + 1

    return index


def _kq_to_num(k, q):
    """
    Converts Rank (k) and order (q) to array index
    Assuming that all ranks are present

    Parameters
    ----------
        k : int
            Rank k
        q : int
            Order q

    Returns
    -------
        int
            Array index
    """

    index = -1
    for kn in range(1, k):
        index += 2*kn + 1
    index += q + k + 1

    return index


K_MAX = 15

stevens_kq_indices = tuple(
        (k, q)
        for k in range(2, K_MAX, 2)
        for q in range(-k, k+1)
)


def perturb_doublets(ener, spin, angm, field=None, verbose=False):
    """Split Kramers doublets along quantisation axis by either applying an
    explicit magnetic field or by rotating each doublet into eigenstates of Jz.

    Parameters
    ----------
    ener : np.array
        Array of SO energies in hartree.
    spin : np.array
        Spin operator in the SO basis.
    angm : np.array
        Orbital angular momentum operator in the SO basis.
    field : float
        Magnetic field in mT applied along the quantisation axis.
    verbose : bool
        Print information about Kramers doublet treatment.

    Returns
    -------
    tuple of np.arrays
        Pair of transformation matrix and its first derivative.
    """

    if not field:  # perturbed J basis for each Kramers doublet

        if verbose:
            print("Rotating Kramers doublets into eigenstates of Jz at zero "
                  "field.")
        
        totj_blks = extract_blocks(spin[2] + angm[2], ener, ener)
        vec = from_blocks(*map(lambda jz: np.linalg.eigh(jz)[1], totj_blks))

    else:  # apply magnetic field

        if verbose:
            print(f"Applying magnetic field of {field} mT to split doublets.")

        zeeman = zeeman_hamiltonian(spin, angm, [0, 0, field])
        _, vec = np.linalg.eigh(np.diag(ener) + zeeman)

    return vec


def print_basis(hamiltonian, spin, angm, space, comp_thresh=0.05, field=None, plot=False, **ops):
    """Print information of basis transformation and plot transformed angmom
    operators.

    Parameters
    ----------
    hamiltonian : np.array
        Array containing the total Hamiltonian in the angmom basis.
    spin : np.array
        Array containing the total spin operator in the angm basis.
    angm : np.array
        Array containing the total orbital angmom operator in the angmom basis.
    space : list of obj
        List of Symbol objects specifing the input space, e.g. terms/levels.
    comp_thresh : float
        Maximum amplitude of a given angular momentum state to be printed in
        the composition section.

    Returns
    -------
    None
    """

    if plot:
        titles = [comp + "-component" for comp in ["x", "y", "z"]]
        ut.plot_op([hamiltonian], 'hamiltonian' + ".pdf")
        for lab, op in ops.items():
            ut.plot_op(op, lab + ".pdf", sq=True, titles=titles)

    # print angmom basis and composition of the electronic states
    basis = space[0].basis

    qn_dict = {op + comp: np.sqrt(
        1/4 + np.diag(cartesian_op_squared(ops[op]).real)) - 1/2
        if comp == '2' else np.diag(ops[op][2]).real
        for op, comp in basis if op in ops and ops[op] is not None}

    def form_frac(rat, signed=True):
        return ('+' if rat > 0 and signed else '') + str(Fraction(rat))

    print("Angular momentum basis:")
    hline = 12 * "-" + "----".join([13 * "-"] * len(basis))

    print(hline)
    print(12 * " " + " || ".join(["{:^13}".format(op) for op in basis]))

    def states():
        for symbol in space:
            for state in symbol.states:
                yield state

    print(hline)
    for idx, state in enumerate(states()):
        print(f"state {idx + 1:4d}: " + " || ".join(
            ["{:>5} ({:5.2f})".format(
                form_frac(getattr(state, op + comp),
                          signed=False if comp == '2' else True),
                qn_dict[op + comp][idx] if op + comp in qn_dict else np.nan)
             for op, comp in basis]))

    print(hline)
    print("Basis labels - state N: [[<theo qn> (<approx qn>)] ...]")

    print()

    print("-----------------------------------------------------------")
    print("Diagonal basis energy, <S_z>, <L_z>, <J_z> and composition:")
    print("( angular momentum kets - " + "|" + ', '.join(basis) + "> )")
    print("-----------------------------------------------------------")

    eig, vec = np.linalg.eigh(hamiltonian)
    eners = (eig - eig[0]) * HARTREE2INVCM

    if field is None:
        vec_total = vec
    else:
        vec_total = vec @ perturb_doublets(eig,
                                           unitary_transform(spin, vec),
                                           unitary_transform(angm, vec),
                                           field=field)

    expectation = zip(*[np.diag(unitary_transform(op[2], vec_total).real)
                        for op in (spin, angm, spin + angm)])

    composition = np.real(vec_total * vec_total.conj())

    def format_state(state, op):
        return form_frac(getattr(state, op), signed=op[1] == 'z')

    for idx, (ener, exp, comp) in enumerate(
            zip(eners, expectation, composition.T), start=1):
        # generate states and sort by amplitude
        super_position = sorted(
            ((amp * 100, tuple(format_state(state, op) for op in basis))
             for state, amp in zip(states(), comp) if amp > comp_thresh),
            key=lambda item: item[0],
            reverse=True)

        print(f"State {idx:4d} {ener:8.2f}  " +
              " ".join([f"{val:+4.2f}" for val in exp]) + " : " +
              '  +  '.join("{:.2f}% |{}>".format(amp, ', '.join(state))
                           for amp, state in super_position))

    print("------------------------")
    print()


def make_operator_storage(op, **ops):

    description_dict = {
        "hamiltonian": "Hamiltonian matrix elements",
        "angm": "Orbital angular momentum matrix elements",
        "spin": "Spin angular momentum matrix elements"
    }

    return StoreOperator(ops[op], op, description_dict[op])


class StoreOperator(Store):

    def __init__(self, op, *args, units='au', fmt='% 20.13e'):

        self.op = op

        super().__init__(*args, label=(), units=units, fmt=fmt)

    def __iter__(self):
        yield (), self.op


class ProjectModelHamiltonian(Store):

    def __init__(self, ops, sf_mult, model_space=None, coupling=None,
                 quax=None, terms=None, k_max=6, theta=None, ion=None,
                 iso_soc=True, verbose=False, units='cm^-1', fmt='% 20.13e'):

        self.ops = ops
        self.sf_mult = sf_mult

        # basis options
        self.model_space = model_space
        self.coupling = coupling
        self.quax = quax

        # model options
        self.terms = terms
        self.theta = theta
        self.ion = ion
        self.iso_soc = iso_soc

        self.verbose = verbose

        description = \
            f"Spin Hamiltonian parameters of the {model_space} multiplet."

        super().__init__('parameters', description,
                         label=("term", "operators"), units=units, fmt=fmt)

    def evaluate(self, **ops):

        term_space, trafo = evaluate_term_space(self.sf_mult, self.model_space,
                                                coupling=self.coupling)

        hamiltonian, spin, angm = \
            evaluate_term_operators(ops['sf_angm'], ops['sf_mch'], ops['sf_amfi'],
                                    self.sf_mult, term_space, quax=self.quax)

        model = SpinHamiltonian(self.model_space, theta=self.theta,
                                ion=self.ion, time_reversal_symm="even",
                                iso_soc=self.iso_soc, **self.terms)

        param_dict = model.project(trafo(hamiltonian), verbose=self.verbose)

        return list(param_dict.values()), list(param_dict.keys())

    def __iter__(self):
        yield from map(lambda val, lab: (self.format_label(lab), val),
                       *self.evaluate(**self.ops))

    def format_label(self, label):
        return (label[0], '_'.join(label[1]))


def terms_by_spin(unique_mults, term_space):
    for mult in unique_mults:
        yield filter(lambda term: term.mult['S'] == mult, term_space)


def evaluate_term_space(sf_mult, model_space, coupling=None, complete=False):

    def reorder_and_expand(cg_vec, term_space):

        # mults, nums = sf_mult
        sf_terms = terms_by_spin(sf_mult.keys(), term_space)

        def reorder(terms):
            try:
                terms, perms = zip(*map(
                    lambda term: term.reduce(ops=('L', 'S'), cls=Term), terms))
                return terms, tuple([idx for perm in perms for idx in perm])
            except ValueError:
                return [], ...

        def pad(vec, mult, num):
            nrows, ncols = vec.shape
            return jnp.vstack([vec, jnp.zeros((num * mult - nrows, ncols))])

        reduced_terms, perms = zip(*map(reorder, sf_terms))
        cg_mults = [term.mult['S'] for term in term_space for _ in term.states]
        cg_vecs = dissect_array(cg_vec, cg_mults, unique=sf_mult, axes=(0,))
        perm_vecs = map(lambda vec, perm: vec[perm, :], cg_vecs, perms)

        if complete:
            # pad with zeros
            padded_vecs = list(map(pad, perm_vecs, sf_mult, sf_mult.values()))
            # stack and complete column space
            vec, rmat = jnp.linalg.qr(jnp.vstack(padded_vecs), mode='complete')
            # re-adjust phases
            vec = vec.at[:, :rmat.shape[1]].multiply(jnp.diag(rmat))
        else:
            vec = jnp.vstack(list(perm_vecs))

        def trafo(op):
            return unitary_transform(op, vec)

        return trafo, [term for terms in reduced_terms for term in terms]

    if coupling:
        warnings.warn("Untested relative phase between different terms.")
        basis_space, cg_vec = couple(model_space, **coupling)
        trafo, term_space = reorder_and_expand(cg_vec.T, basis_space)

    elif isinstance(model_space, Level):
        basis_space = [Term(L=model_space.qn['L'], S=model_space.qn['S'])]
        _, cg_vec = basis_space[0].couple('J', 'L', 'S', levels=[model_space])
        trafo, term_space = reorder_and_expand(cg_vec, basis_space)

    elif isinstance(model_space, Term):
        # trafo, term_space = lambda x: x, [model_space]
        trafo, term_space = reorder_and_expand(
            np.identity(model_space.multiplicity), [model_space])

    else:
        raise ValueError(f"Invalid model_space {model_space}!")

    return term_space, trafo


def evaluate_sf_operators(angm, ener, amfi, spin_mult):

    sf_mult = dict(zip(*np.unique(spin_mult, return_counts=True)))

    sf_angm = list(extract_blocks(angm, spin_mult, spin_mult))
    sf_amfi = list(map(list, dissect_array(amfi, spin_mult, spin_mult)))
    sf_ener = list(map(jnp.diag, extract_blocks(ener, spin_mult)))

    return sf_angm, sf_ener, sf_amfi, sf_mult


def evaluate_term_operators(sf_angm, sf_ener, sf_amfi, sf_mult, term_space,
                            quax=None, complete=False):

    sf_terms = list(map(list, terms_by_spin(sf_mult, term_space)))

    def amfi_phase(amfi_blks):

        def relative_phase(blk):
            return np.mean(np.angle(blk))# - np.pi

        def qns(term):
            for proj in np.arange(-term.qn['L'], term.qn['L'] + 1):
                yield term.qn['L'], proj

        for rmult, row, rterms in zip(sf_mult, amfi_blks, sf_terms):
            for cmult, blk, cterms in zip(sf_mult, row, sf_terms):
                if cmult - rmult == 2 and rterms and cterms:
                    amfi_1 = 1.j * (blk[0] + 1.j * blk[1])
                    mask = jnp.nonzero(jnp.array(
                        [[True if rl2 == cl2 and rlz == clz + 1 else False
                          for cterm in cterms for cl2, clz in qns(rterm)]
                         for rterm in rterms for rl2, rlz in qns(rterm)]))

                    yield jnp.exp(1.j * relative_phase(amfi_1[mask]))
                    break
            else:
                yield 1.0

    # Rotation of quantisation axis
    if quax is not None:
        sf_angm = sfy(lambda x: rotate_cart(x, quax), sf=1)(sf_angm)
        sf_amfi = sfy(lambda x: rotate_cart(x, quax), sf=2)(sf_amfi)

    sf_vecs = [project_angm_basis(iter(terms), a, complete=complete)
               for terms, a in zip(sf_terms, sf_angm)]

    # fix relative phase inter-term phase
    term_amfi = unitary_transform(sf_amfi, sf_vecs, sf=2)
    relative_phases = amfi_phase(term_amfi)
    sf_vecs = [ph * vec for ph, vec in zip(relative_phases, sf_vecs)]

    term_ener = unitary_transform(sf_ener, sf_vecs, sf=1)
    term_amfi = unitary_transform(sf_amfi, sf_vecs, sf=2)
    term_angm = unitary_transform(sf_angm, sf_vecs, sf=1)
    # amfi_full = 1.j * np.block(term_amfi)
    # ut.plot_op(amfi_full, "amfi.pdf")
    # ut.plot_op(from_blocks(*term_angm), "angm.pdf")

    if complete:
        term_smult = np.repeat(list(sf_mult.keys()), list(sf_mult.values()))
    else:
        term_smult = [term.mult['S'] for terms in sf_terms for term in terms
                      for _ in range(term.mult['L'])]

    ws_angm = sf2ws(term_angm, sf_mult)
    ws_spin = np.array(make_angmom_ops_from_mult(term_smult)[0:3])
    ws_hamiltonian = sf2ws(term_ener, sf_mult) + sf2ws_amfi(term_amfi, sf_mult)

    return ws_hamiltonian, ws_spin, ws_angm


def read_params(file, group='/', **mapping):
    with h5py.File(file, 'r') as h:
        for term in iter(grp := h[group]):
            if term == 'diag':
                pass
            else:
                for ops in iter(grp[term]):
                    path = grp['/'.join([term, ops, 'parameters'])]
                    op_labs = tuple(mapping.get(o, o) for o in ops.split('_'))
                    key = path.attrs['typename']
                    env = {key: namedtuple(key, path.attrs['field_names'])}
                    names = [eval(row, env) for row in path.attrs['row_names']]
                    data = path[...]
                    yield (term, op_labs), {k: v for k, v in zip(names, data)}


class SpinHamiltonian:
    """Set up model spin Hamiltonian to be fitted to ab initio Hamiltonian in
    angular momentum basis.
    The model might be composed of: H = V_0 + H_so + H_ex + H_cf + H_zee
    (V_0: diagonal shift, H_so: spin-orbit coupling, H_ex: exchange coupling,
    H_cf: CF interaction, H_zee: Zeeman effect).

    Parameters
    ----------
    symbol : obj
        Symbol object specifying the angular momentum space.
    angm_ops : dict, default = None
        Dictionary of angm operators. Keys are the angm operator labels. If
        omitted, exact operators are used.
    k_max : int, default = 6
        Maximum Stevens operator rank used in crystal field Hamiltonian.
    theta : bool, default = False
        If True, factor out operator equivalent factors theta.
    diag : bool
        If True, include constant diagonal shift term.
    iso_soc : bool
        If True, SOC interaction is described by isotropic operator.
    time_reversal_symm : ["even", "odd"], default "even"
        If "even" ("odd"), only include exchange terms which are "even" ("odd")
        under time reversal.
    ion : object, default = None
        Ion object for operator equivalent factor lookup.
    **terms: keyword arguments
        Terms to include in the model Hamiltonian specified as:
            spin-orbit coupling: soc=[("L", "S")]
            crystal field: cf=[("L",)]
            exchange: ex=[("R", "S"), ("R", "L"), ("R", "S", "L")]
            Zeeman: Lzee=[("L",)] Szee=[("s", "R")]


    Attributes
    ----------
    symbol : obj
        Symbol object specifying the angular momentum space.
    angm : dict
        Dictionary of angm operators. Keys are the angm operator labels.
    k_max : int
        Maximum Stevens operator rank used in crystal field Hamiltonian.
    theta : bool, default = False
        If true, factor out operator equivalent factors theta.
    ion : object, default = None
        Ion object for operator equivalent factor lookup.
    term_dict : dict of dicts
        Dictionary of terms. Each entry of sub-dict is a contribution to the
        model Hamiltonian associated with a parameter.
    term_len : dict
        Dictionary of number of parameter of each term in model Hamiltonian.
    """

    def __init__(self, symbol, angm_ops=None, ion=None, k_max=6, theta=False,
                 diag=True, iso_soc=True, time_reversal_symm="even", **terms):

        self.symbol = symbol
        self.ion = ion

        self.angm = \
            {o: tuple(angm_ops[o]) + (angm_ops[o][0] + 1.j * angm_ops[o][1],
                                      angm_ops[o][0] - 1.j * angm_ops[o][1],
                                      cartesian_op_squared(angm_ops[o])[0])
             if angm_ops is not None and o in angm_ops else
             calc_ang_mom_ops(qn) for o, qn in self.symbol.qn.items()}

        self.k_max = k_max
        self.theta = theta
        self.iso_soc = iso_soc
        self.time_reversal_symm = time_reversal_symm
        self.diag = diag

        print(f"Including time-reversal {self.time_reversal_symm} terms.")

        self.terms = ({"diag": [()]} if diag else {}) | terms

    def __iter__(self):

        resolve = {
            "diag": self._build_diag,
            "soc": self._build_soc,
            "cf": self._build_cf,
            "ex": self._build_ex,
            "Lzee": partial(self._build_zee, orbital=True),
            "Szee": partial(self._build_zee, orbital=False)
        }

        return (((term, labs), resolve[term](labs))
                for term, sub in self.terms.items() for labs in sub)

    def print_basis(self):
        print(self.symbol.states)

    def _build_diag(self, ops):
        if ops:
            raise ValueError("Inconsistency in building diagonal shift op.")

        Key = namedtuple('shift', '')
        return ((Key(), jnp.identity(self.symbol.multiplicity)),)

    def _build_soc(self, ops):

        if self.iso_soc:
            Key = namedtuple('lamb', '')
            return ((Key(), jnp.sum(jnp.array([
                reduce(jnp.kron,
                       [self.angm[o][c] if o in ops else jnp.identity(m)
                        for o, m in self.symbol.mult.items()])
                for c in range(3)]), axis=0)),)
        else:
            Key = namedtuple('lamb', 'component')
            return ((Key(("x", "y", "z")[c - 1]),
                    reduce(jnp.kron,
                           [self.angm[o][c] if o in ops else jnp.identity(m)
                            for o, m in self.symbol.mult.items()]))
                    for c in range(3))

    def _build_cf(self, ops):

        op = ops[0]

        Okq = \
            calc_stev_ops_jax(self.k_max, (self.symbol.mult[op] - 1) / 2,
                          self.angm[op][3], self.angm[op][4], self.angm[op][2])

        if not self.theta:
            pass
        elif self.theta and op.upper() in ANGM_SYMBOLS:
            theta = self.ion.theta('l')
        elif self.theta and op.upper() in TOTJ_SYMBOLS:
            theta = self.ion.theta('j')
        else:
            raise ValueError(f"Unknown angular momentum identifier: {op}")

        Key = namedtuple('B', 'k q')
        return ((Key(k, q),
                reduce(jnp.kron,
                       [Okq[k - 1, k + q, ...] *
                        (theta[k] if self.theta else 1.0)
                        if o == op else jnp.identity(m)
                        for o, m in self.symbol.mult.items()]))
                for k in range(2, self.k_max + 1, 2) for q in range(-k, k + 1))

    def _build_zee(self, ops, orbital=False):
        muB = 0.5  # atomic units
        au2mT = 2.35051756758e5 * 1e3  # mTesla / au
        g_e = 2.002319

        Key = namedtuple('B', 'comp')
        return ((Key(comp),
                 HARTREE2INVCM * muB / au2mT * (1.0 if orbital else g_e) *
                 np.sum(
                     [reduce(np.kron,
                             [self.angm[o][comp] if o == op else np.identity(m)
                              for o, m in self.symbol.mult.items()])
                      for op in ops],
                     axis=0))
                for comp in range(3))

    def _build_ex(self, ops):

        def time_rev_symm(ranks):
            if self.time_reversal_symm == "even":
                return not sum(ranks) % 2
            elif self.time_reversal_symm == "odd":
                return sum(ranks) % 2
            else:
                return True

        Okqs = {o: calc_stev_ops_jax(
            self.symbol.mult[o] - 1, self.symbol.qn[o],
            self.angm[o][3], self.angm[o][4], self.angm[o][2]) for o in ops}

        kdc = (dict(zip(ops, idc))
               for idc in product(*(range(1, self.symbol.mult[o])
                                    for o in ops)))

        # generator of orders
        def qdc(kdx):
            return (dict(zip(ops, idc))
                    for idc in product(
                        *(range(-k, k + 1) for k in kdx.values())))

        idc = iter([('k', 'q'), ('n', 'm')])
        Key = namedtuple('J', (i for o in ops for i in (("alpha",) if o == 'R'
                                                        else next(idc))))

        return ((Key(*(i for o, kx, qx in zip(ops, k.values(), q.values())
                for i in ((('z', 'x', 'y')[qx],) if o == 'R' else (kx, qx)))),
                (-1) * reduce(jnp.kron,
                              [Okqs[o][k[o] - 1, k[o] + q[o], ...] /
                               (1.0 if o.upper() == 'R' else
                                Okqs[o][k[o] - 1, k[o], -1, -1])  # IC scalar
                               if o in ops else jnp.identity(m)
                               for o, m in self.symbol.mult.items()]))
                for k in kdc for q in qdc(k) if time_rev_symm(k.values()))

    def project(self, H_ai, verbose=False):
        """Project ab initio Hamiltonian onto model.

        Parameters
        ----------
        H_ai : np.array
            Ab initio Hamiltonian in the appropiate basis. (Ordering according
            to basis_mult argument of constructor.)
        verbose : bool
            Flag for printing information from least squares fit and plot
            original and fitted Hamiltonian matrices.

        Returns
        -------
        dict of dicts
            Dictionary of terms. Each term is a dictionary itself listing all
            projected model parameters. Sub-keys are Stevens operator rank
            order pairs in the same order as defined in the **terms parameters.
        """

        H_ai = H_ai * HARTREE2INVCM

        def proj(op):
            return jnp.sum(H_ai * op.conj()).real / jnp.linalg.norm(op)**2

        # def orthonorm(op1, op2):
        #     return (np.sum(op1 * op2.conj()) / (np.linalg.norm(op1) * np.linalg.norm(op2))).real

        params = {(term, labs): {key: proj(op) for key, op in ops}
                  for (term, labs), ops in iter(self)}

        # print(np.array([[orthonorm(op1, op2) for _, ops1 in iter(self) for _, op1 in ops1] for _, ops2 in iter(self) for _, op2 in ops2]))

        H_fit = self.parametrise(params, verbose=False)
        err = jnp.linalg.norm(H_fit - H_ai)**2

        if verbose:
            print("Absolute err (RMSD, i.e. sqrt[1/N^2 * sum of squared "
                  "residuals])\n{:10.4f}".format(
                      jnp.sqrt(err / H_ai.size)))
            print("Relative err (sqrt[sum of squared residuals] / "
                  "norm of ab initio Hamiltonian)\n{:10.4%}".format(
                      jnp.sqrt(err) / jnp.linalg.norm(H_ai)))

            print("Eigenvalues of the ab initio and model Hamiltonian "
                  "(diagonal shift substracted):")

            shift = list(params[("diag", ())].values())[0] if self.diag else 0.
            diag_shift = shift * jnp.identity(self.symbol.multiplicity)
            eig_a, _ = jnp.linalg.eigh(H_ai - diag_shift)
            eig_m, _ = jnp.linalg.eigh(H_fit - diag_shift)

            for i, (a, m) in enumerate(zip(eig_a, eig_m), start=1):
                print(f"{i} {a} {m}")

            ut.plot_op([H_ai, H_fit], "h_ai.pdf",
                       titles=["Ab initio Hamiltonian", "Model fit"])

        return params

    def parametrise(self, params, scale=None, verbose=False):

        ham = jnp.zeros((self.symbol.multiplicity, self.symbol.multiplicity),
                        dtype='complex128')

        for lab, ops in iter(self):
            for key, op in ops:
                if verbose:
                    print(f"Parametrising {key} of {lab[0]}{lab[1]}")
                if scale is None:
                    ham += params[lab][key] * op
                else:
                    ham += params[lab][key] * op * scale.get(lab[0], 1.0)

        return ham
        # return reduce(lambda x, y: x + y,
        #               (params[lab][key] * op
        #                for lab, ops in iter(self) for key, op in ops))


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

    def evaluate(self, **ops):

        if self.differential:
            tensor_func = partial(susceptibility_tensor,
                                  hamiltonian=ops['hamiltonian'],
                                  spin=ops['spin'], angm=ops['angm'],
                                  field=self.field,
                                  differential=self.differential)
        else:
            tensor_func = make_susceptibility_tensor(
                hamiltonian=ops['hamiltonian'],
                spin=ops['spin'], angm=ops['angm'],
                field=self.field)

        if self.iso:
            def func(temp):
                return jnp.trace(tensor_func(temp)) / 3
        else:
            func = tensor_func

        # vmap does not repeat the eigen decomp
        if self.differential:
            chi_list = [func(temp) for temp in self.temperatures]
        else:  # non-bached more efficient when using the expm backend
            chi_list = [func(temp) for temp in self.temperatures]
            # chi_list = vmap(func)(jnp.array(self.temperatures))

        Key = namedtuple('chi', 'temp')
        data = {Key(temp): (temp * chi) if self.chi_T else chi
                for temp, chi in zip(self.temperatures, chi_list)}
        return [data], [()]

    def __iter__(self):
        yield from ((lab, dat) for dat, lab in zip(*self.evaluate(**self.ops)))


class MagneticSusceptibilityFromFile(FromFile, MagneticSusceptibility):
    pass


class EPRGtensor(Store):

    def __init__(self, ops, eprg_values=False, eprg_vectors=False,
                 eprg_tensors=False, units='au', fmt='% 20.13e'):

        self.ops = ops

        # basis options
        self.eprg_values = eprg_values
        self.eprg_vectors = eprg_vectors
        self.eprg_tensors = eprg_tensors

        if self.eprg_values:
            args = ("eprg_values", "Principal values of the EPR G-tensor")
        elif self.eprg_vectors:
            args = ("eprg_vectors", "Principal axes of the EPR G-tensor")
        elif self.eprg_tensors:
            args = ("eprg_tensors", "EPR G-tensor")
        else:
            raise ValueError("Supply one of eprg_values/_vectors/_tensors!")

        super().__init__(*args, label=("doublet",), units=units, fmt=fmt)

    def evaluate(self, **ops):

        eig, vec = jnp.linalg.eigh(ops['hamiltonian'])
        labs = np.unique(np.around(eig, 8), return_inverse=True)[1]
        spin_blks = extract_blocks(unitary_transform(ops['spin'], vec), labs, labs)
        angm_blks = extract_blocks(unitary_transform(ops['angm'], vec), labs, labs)
        eprg_list = map(eprg_tensor, spin_blks, angm_blks)

        if self.eprg_tensors:
            data = list(eprg_list)

        else:
            eprg_vals, eprg_vecs = zip(*map(jnp.linalg.eigh, eprg_list))

            if self.eprg_values:
                data = eprg_vals
            elif self.eprg_vectors:
                data = eprg_vecs

        return list(data), [(idx,) for idx, _ in enumerate(data, start=1)]

    def __iter__(self):
        yield from ((lab, dat) for dat, lab in zip(*self.evaluate(**self.ops)))


class EPRGtensorFromFile(FromFile, EPRGtensor):
    pass


class Tint(Store):

    def __init__(self, ops, field=0., states=None, units='au', fmt='% 20.13e'):

        self.ops = ops
        self.field = field
        self.states = states

        super().__init__(
            "tint",
            "Matrix elements of the magnetic dipole transition intensity",
            label=("istate",), units=units, fmt=fmt)

    def evaluate(self, **ops):

        zee = zeeman_hamiltonian(
                ops['spin'], ops['angm'], np.array([0., 0., self.field]))
        _, vec = jnp.linalg.eigh(ops['hamiltonian'] + zee)

        vec_out = vec if self.states is None else vec[:, list(self.states)]

        magm = vec_out.conj().T @ magmom(ops['spin'], ops['angm']) @ vec
        tint = np.sum(np.real(magm * magm.conj()), axis=0) / 3

        Key = namedtuple('jstate', 'index')
        data = [{Key(idx): val for idx, val in enumerate(row, start=1)} for row in tint]
        return data, [(idx,) for idx, _ in enumerate(data, start=1)]

    def __iter__(self):
        yield from ((lab, dat) for dat, lab in zip(*self.evaluate(**self.ops)))

    def __iter__(self):
        yield from ((lab, dat) for dat, lab in zip(*self.evaluate(**self.ops)))


class TintFromFile(FromFile, Tint):
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
    return jnp.einsum('i,imn->mn', jnp.array(field) / au2mT, magmom(spin, angm))


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


def molecular_magnetisation(temp, hamiltonian, spin, angm, field, algorithm='eigh'):
    """ Molar molecular magnetisation in [hartree] / [mT mol]

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
    field : np.array
        Magnetic field in mT at which susceptibility is measured. If None,
        returns differential susceptibility.
    algorithm : {'eigh', 'expm'}
        Algorithm for the computation of the partition function.
    """

    Na = 6.02214076e23  # 1 / mol
    kb = 3.166811563e-6  # hartree / K
    beta = 1 / (kb * temp)  # hartree
    au2mT = 2.35051756758e5 * 1e3  # mTesla / au

    h_total = hamiltonian + zeeman_hamiltonian(spin, angm, field)

    if algorithm == 'expm':
        dim = h_total.shape[0]
        # condition matrix by diagonal shift
        h_shft = h_total - stop_gradient(jnp.eye(dim) * jnp.min(h_total))
        expH = jscipy.linalg.expm(-beta * h_shft)
        Z = jnp.trace(expH).real

    elif algorithm == 'eigh':
        eig, vec = jnp.linalg.eigh(h_shft)
        eig_shft = eig - stop_gradient(eig[0])
        expH = vec @ jnp.diag(jnp.exp(-beta * eig_shft)) @ vec.T.conj()
        Z = jnp.sum(jnp.exp(-beta * eig_shft))

    else:
        ValueError(f"Unknown algorithm {algorithm}!")

    dZ = -jnp.einsum('ij,mji', expH, magmom(spin, angm) / au2mT).real

    return Na * dZ / Z


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

    Na = 6.02214076e23  # 1 / mol
    kb = 3.166811563e-6  # hartree / K
    au2mT = 2.35051756758e5 * 1e3  # mTesla / au

    h_total = hamiltonian + zeeman_hamiltonian(spin, angm, field)
    # condition matrix by diagonal shift
    eig, vec = jnp.linalg.eigh(h_total)

    def molecular_magnetisation(temp):
        beta = 1 / (kb * temp)  # hartree
        eig_shft = eig - stop_gradient(eig[0])
        expH = vec @ jnp.diag(jnp.exp(-beta * eig_shft)) @ vec.T.conj()
        Z = jnp.sum(jnp.exp(-beta * eig_shft))
        dZ = -jnp.einsum('ij,mji', expH, magmom(spin, angm) / au2mT).real
        return Na * dZ / Z

    return molecular_magnetisation
