import re
from itertools import product, chain
from functools import reduce, lru_cache
import itertools
from fractions import Fraction
from collections import namedtuple

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from jax import numpy as jnp, grad, jacfwd, core, ad, lax
from jax.scipy.linalg import block_diag, expm, qr

from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_3j
import utils as ut
from group import project_angm


SPIN_SYMBOLS = ['R', 'S', 'T', 'U', 'V', 'W']
ANGM_SYMBOLS = ['L', 'M', 'N', 'O', 'P', 'Q']
TOTJ_SYMBOLS = ['F', 'G', 'H', 'I', 'J', 'K']


def sf2ws(sf_op, sf_mult):

    def expand(op, mult):
        return jnp.kron(op, np.identity(mult))

    return from_blocks(*map(expand, sf_op, sf_mult))


def sf2ws_spin(sf_op, sf_smult):

    def me(s1, ms1, sf1, s2, ms2, sf2):
        if jnp.abs(s1 - s2) <= 2 and jnp.abs(ms1 - ms2) <= 2:
            return coeff(s1, ms1, s2, ms2) * op(s1, ms1, sf1, s2, ms2, sf2)
        else:
            return 0.0

    def op(s1, ms1, sf1, s2, ms2, sf2):

        if jnp.abs(s1 - s2) <= 2:
            if ms1 == ms2:
                return sf_op[2, sf1, sf2]
            elif ms1 + 2 == ms2:
                return (+ sf_op[0, sf1, sf2] + 1.j * sf_op[1, sf1, sf2])
            elif ms1 - 2 == ms2:
                return (- sf_op[0, sf1, sf2] + 1.j * sf_op[1, sf1, sf2])
            else:
                return 0.0
        else:
            return 0.0

    def coeff(s1, ms1, s2, ms2):
        # double integer figures and extra "/ 2" factor in common factor due 2
        # double quantum number convention

        if s1 == s2:
            if s1 == 0:
                return 0.0
            elif ms1 == ms2:
                c = ms1
            elif ms1 + 2 == ms2:
                c = + jnp.sqrt((s1 - ms1) * (s1 + ms1 + 2)) / 2
            elif ms1 - 2 == ms2:
                c = - jnp.sqrt((s1 + ms1) * (s1 - ms1 + 2)) / 2
            else:
                c = 0.0
            return c / jnp.sqrt(s1 * (s1 + 2) * (2 * s1 + 2) / 2)

        elif s1 + 2 == s2:
            if ms1 == ms2:
                c = jnp.sqrt((s1 + 2)**2 - ms1**2)
            elif ms1 + 2 == ms2:
                c = - jnp.sqrt((s1 + ms1 + 2) * (s1 + ms1 + 4)) / 2
            elif ms1 - 2 == ms2:
                c = - jnp.sqrt((s1 - ms1 + 2) * (s1 - ms1 + 4)) / 2
            else:
                c = 0.0
            return c / jnp.sqrt((s1 + 1) * (2 * s1 + 1) * (2 * s1 + 3) / 2)
        
        elif s1 - 2 == s2:
            if ms1 == ms2:
                c = jnp.sqrt(s1**2 - ms1**2)
            elif ms1 + 2 == ms2:
                c = jnp.sqrt((s1 - ms1) * (s1 - ms1 - 2)) / 2
            elif ms1 - 2 == ms2:
                c = jnp.sqrt((s1 + ms1) * (s1 + ms1 - 2)) / 2
            else:
                c = 0.0
            return c / jnp.sqrt(s1 * (2 * s1 - 1) * (2 * s1 + 1) / 2)

        else:
            return 0.0

    if sf_op is None:
        ws_op = None

    else:
        ws_op = jnp.array([
            [me(s1, ms1, sf1, s2, ms2, sf2) for s2, ms2, sf2 in zip(
                [m - 1 for m in sf_smult for _ in range(m)],
                [- (m - 1) + 2 * i for m in sf_smult for i in range(m)],
                [i for i, m in enumerate(sf_smult) for _ in range(m)])
             ] for s1, ms1, sf1 in zip(
                [m - 1 for m in sf_smult for _ in range(m)],
                [- (m - 1) + 2 * i for m in sf_smult for i in range(m)],
                [i for i, m in enumerate(sf_smult) for _ in range(m)])
            ])

    return ws_op


def sf2ws_amfi(sf_op, sf_mult):

    @lru_cache
    def coeff(q, s1, s2):
        # extra minus sign to ensure hermiticity
        return jnp.array([[(-1)**(s2 + m1 - 1) * (-1.0 if s2 > s1 else 1.0) *
                           float(wigner_3j(s2, 1, s1, m2, q, -m1))
                           for m2 in jnp.arange(-s2, s2 + 1, 1)]
                          for m1 in jnp.arange(-s1, s1 + 1, 1)])

    def expand(op, mult1, mult2):

        s1, s2 = (mult1 - 1) / 2, (mult2 - 1) / 2

        if abs(s1 - s2) <= 1:
            spherical_components = jnp.array([
                jnp.kron(op[0] + 1.j * op[1], coeff(-1, s1, s2)) / jnp.sqrt(2),
                jnp.kron(op[2], coeff(0, s1, s2)),
                -jnp.kron(op[0] - 1.j * op[1], coeff(+1, s1, s2)) / jnp.sqrt(2)
            ])

            return jnp.sum(spherical_components, axis=0)
        else:
            return jnp.zeros((mult1 * op.shape[1], mult2 * op.shape[2]))

    # introduce imaginary unit
    return jnp.block([[expand(1.j * op, mult1, mult2)
                       for mult2, op in zip(sf_mult, row)]
                      for mult1, row in zip(sf_mult, sf_op)])


def sfy(fun, sf=0):
    if sf == 0:
        return fun
    else:
        return lambda *x: list(map(lambda *y: sfy(fun, sf=sf - 1)(*y), *x))


def unitary_transform(op, Umat, sf=0):
    if sf == 0:
        return Umat.conj().T @ op @ Umat
    elif sf == 1:
        return list(map(unitary_transform, op, Umat))
    elif sf == 2:
        return [[u.conj().T @ o @ w for o, w in zip(row, Umat)]
                for row, u in zip(op, Umat)]
    else:
        raise ValueError(f"Invalid sf={sf} argument.")


def cartesian_op_squared(op):
    return jnp.sum(op @ op, axis=0)


def rotate_cart(op, rot_mat):
    return np.einsum('ji,imn->jmn', rot_mat, op)


def dissect_array(mat, *blk_labs, unique=None, axes=None):
    ndim = len(mat.shape)
    for ulab in np.unique(blk_labs[0], axis=0) if unique is None else unique:

        slice_idc = np.flatnonzero([lab == ulab for lab in blk_labs[0]])

        if axes is None:
            idc = tuple(slice_idc if ndim - i == len(blk_labs) else slice(None)
                        for i in range(ndim))
        else:
            idc = tuple(slice_idc if i == axes[0] else slice(None)
                        for i in range(ndim))

        _mat = mat[idc]

        if len(blk_labs) > 1:
            yield dissect_array(_mat, *blk_labs[1:], unique=unique, axes=axes)
        else:
            yield _mat


def extract_blocks(mat, *blk_labs, unique=None):
    for ulab in np.unique(blk_labs[0], axis=0) if unique is None else unique:
        idc = [np.flatnonzero([lab == ulab for lab in labs])
               for labs in blk_labs]
        yield mat[(...,) + np.ix_(*idc)]


def from_blocks(*blocks):
    """Generalisation of scipy.block_diag for multi-component operators.
    """
    try:
        return block_diag(*blocks)
    except ValueError:
        return jnp.array([from_blocks(*comps) for comps in zip(*blocks)])


def phase(op, sgn="pos"):

    angles = jnp.angle(jnp.diag(op, k=-1))

    Amat = jnp.diag(jnp.concatenate([-jnp.ones(angles.size)]), k=1) + \
        jnp.diag(jnp.ones(angles.size + 1))

    phase_ang = jnp.linalg.solve(Amat, jnp.append(-angles, 0))

    return jnp.diag(jnp.exp(1.j * phase_ang))


def calc_ang_mom_ops(J):
    """
    Calculates the angular momentum operators jx jy jz jp jm j2

    Parameters
    ----------
    J : float
        J quantum number

    Returns
    -------
    np.array
        Matrix representation of jx angular momentum operator
    np.array
        Matrix representation of jy angular momentum operator
    np.array
        Matrix representation of jz angular momentum operator
    np.array
        Matrix representation of jp angular momentum operator
    np.array
        Matrix representation of jm angular momentum operator
    np.array
        Matrix representation of j2 angular momentum operator
    """

    # Create vector of mj values
    mj = np.arange(-J, J + 1, 1, dtype=np.complex128)

    # jz operator - diagonal in jz basis- entries are mj
    jz = np.diag(mj)

    # jp and jm operators
    jp = np.array([[np.sqrt(J * (J + 1) - m2 * (m2 + 1)) if m1 == m2+1 else 0.0
                    for it2, m2 in enumerate(mj)]
                   for it1, m1 in enumerate(mj)])

    jm = np.array([[np.sqrt(J * (J + 1) - m2 * (m2 - 1)) if m1 == m2-1 else 0.0
                    for it2, m2 in enumerate(mj)]
                   for it1, m1 in enumerate(mj)])

    # jx, jy, and j2
    jx = 0.5 * (jp + jm)
    jy = 1. / (2. * 1j) * (jp - jm)
    j2 = jx @ jx + jy @ jy + jz @ jz

    return jx, jy, jz, jp, jm, j2


def make_angmom_ops_from_mult(mult):
    """
    Calculates the angular momentum operators jx jy jz jp jm j2 for a manifold
    of multiplicities. The resulting operator take block diagonal shape.

    Parameters
    ----------
    mult : list
        Array of multiplicities.

    Returns
    -------
    np.array
        Matrix representation of jx angular momentum operator
    np.array
        Matrix representation of jy angular momentum operator
    np.array
        Matrix representation of jz angular momentum operator
    np.array
        Matrix representation of jp angular momentum operator
    np.array
        Matrix representation of jm angular momentum operator
    np.array
        Matrix representation of j2 angular momentum operator
    """

    j = np.block([[
        np.array(calc_ang_mom_ops((smult1 - 1) / 2)[0:3]) if idx1 == idx2 else
        np.zeros((3, smult1, smult2))
        for idx2, smult2 in enumerate(mult)]
        for idx1, smult1 in enumerate(mult)])

    j2 = cartesian_op_squared(j)

    return j[0], j[1], j[2], j[0] + 1.j * j[1], j[0] - 1.j * j[1], j2


def print_sf_term_content(sf_angm, sf_eners, spin_mult, max_angm=13):

    term_composition = \
        [{Term(L=qn, S=(mult - 1) / 2): np.diag(project_angm(qn, cartesian_op_squared(angm))).real
                for qn in range(max_angm)}
         for mult, angm in zip(spin_mult, sf_angm)]

    print("Spin-free section:")
    for mult, composition, eners in zip(spin_mult, term_composition, sf_eners):
        print(f"S = {(mult-1)/2}")
        print_terms(composition, eners)


def print_so_term_content(so_spin, so_angm, eners, spin_mults, max_angm=13):

    def mult2qn(mult):
        return (mult - 1) / 2

    def make_proj(j_ops):
        @lru_cache
        def partial_func(qn):
            return project_angm(qn, cartesian_op_squared(j_ops))
        return partial_func

    proj_spin, proj_angm, proj_totj = \
        map(make_proj, [so_spin, so_angm, so_spin + so_angm])

    term_composition = \
        {Level(S=spin_qn, L=angm_qn, J=totj_qn, coupling={'J': (('L', None), ('S', None))}):
         np.diag(proj_totj(totj_qn) @ proj_angm(angm_qn) @ proj_spin(spin_qn)).real
         for spin_qn in map(mult2qn, spin_mults)
         for angm_qn in range(max_angm)
         for totj_qn in np.arange(np.abs(angm_qn - spin_qn), angm_qn + spin_qn + 1)}

    print("Spin-orbit section:")
    print_terms(term_composition, eners)


def print_terms(composition, eners):

    print("=" * 33)
    print("Term composition:")
    print("-" * 33)
    for term, comp in composition.items():
        content = ' + '.join([f"{contr:4.2f}|{idx}>" for idx, contr in
                              enumerate(comp, start=1) if contr > 0.1])
        count = np.sum(comp) / (term.mult['J'] if isinstance(term, Level) else term.mult['L'])
        if count > 0.1:
            print(f"{count:4.2f} |{term}> = {content}")
    print("=" * 33)
    print("State composition:")
    print("-" * 33)
    for state, ener in enumerate(eners):
        content = ' + '.join([f"{comp[state]:4.2f}|{term}>"
                              for term, comp in composition.items()
                              if comp[state] > 0.1])
        print(f"|State {state+1:3d}> ({ener:8.2f}) = {content}")
    print("=" * 33)


def block_triangular_expm(A, B, C, t):
    n = A.shape[0]
    idc = np.ix_(range(0, n), range(n, 2 * n))
    blk = jnp.block([[A, B], [jnp.zeros((n, n)), C]])
    return expm(t * blk)[idc]


def integrate_expm(A, t):
    n = A.shape[0]
    return block_triangular_expm(A, jnp.identity(n), jnp.zeros((n, n)), t)


def integrate_cosk_expm(k, A, t):
    n = A.shape[0]
    return integrate_expm(A + 1.j * k * np.identity(n), t).real


def integrate_SO3_char_psi(j, Jz):
    return 0.5 * (integrate_cosk_expm(j, Jz, np.pi) -
                  integrate_cosk_expm(j + 1, Jz, np.pi))


def integrate_SO3_theta(Jy, B):
    dim = Jy.shape[0]
    return block_triangular_expm(
        Jy, B, Jy + 1.j * jnp.identity(dim), np.pi).imag @ expm(-np.pi * Jy)


def integrate_SO3_phi(Jz, B):
    return block_triangular_expm(Jz, B, Jz, 2 * np.pi) @ expm(-2 * np.pi * Jz)


def count_SO3_irrep(j, Jz):
    return 2/np.pi * jnp.trace(integrate_SO3_char_psi(j, Jz))


def project_SO3_irrep(j, Jy, Jz):
    psi_int = integrate_SO3_char_psi(j, Jz)
    theta_int = integrate_SO3_theta(Jy, psi_int)
    phi_int = integrate_SO3_phi(Jz, theta_int)
    return (2 * j + 1) / (2 * np.pi**2) * phi_int


def project_angm_basis(terms, angm, complete=False):

    try:
        # orthogonal projection operator into subspace of irrep j
        j = next(terms).qn['L']
        proj = project_SO3_irrep(j, angm[1].imag, angm[2].imag)
        _, _, perm = scipy.linalg.qr(proj, pivoting=True)
        # proj_vec, r = qr(proj[:, jnp.argsort(-jnp.diag(proj))])
        proj_vec, r = qr(proj[:, perm])

        # split basis of irrep space from its orthogonal complement
        irrep_vec, compl_vec = proj_vec[:, :(2*j + 1)], proj_vec[:, (2*j + 1):]

        # diagonalise z-component + phase x-component
        _, diag_vec = jnp.linalg.eigh(unitary_transform(angm[2], irrep_vec))

        ph_vec = phase(unitary_transform(angm[0], irrep_vec @ diag_vec))

        irrep_basis = irrep_vec @ diag_vec @ ph_vec
        compl_basis = project_angm_basis(
            terms, unitary_transform(angm, compl_vec), complete=complete)

        return jnp.hstack([irrep_basis, compl_vec @ compl_basis])

    except StopIteration:
        if complete:
            return jnp.identity(angm.shape[1])
        else:
            return jnp.empty((angm.shape[1], 0))


class Symbol:

    def __init__(self, coupling=None, **qn):

        if coupling is None:
            self.coupling = {op: None for op in qn}
        else:
            self.coupling = coupling

        self.qn = qn

        self.mult = {op: int(2 * qn) + 1 for op, qn in qn.items()
                     if op in self.coupling}

        self.multiplicity = reduce(lambda x, y: x * y, self.mult.values())

        self.basis = [op + comp for op in self.qn for comp in ('2', 'z')
                      if comp == '2' or op in self.coupling]

    def elementary_ops(self, group=None):

        if not (group == 'spin' or group == 'angm' or group is None):
            raise ValueError(f'Invalid group argument {group}!')

        def generate_leaves(op):
            match op:
                case (j, None):
                    if group is None:
                        yield j
                    elif group == 'spin' and j in SPIN_SYMBOLS:
                        yield j
                    elif group == 'angm' and j in ANGM_SYMBOLS:
                        yield j

                case (j, (j1, j2)):
                    yield from generate_leaves(j1)
                    yield from generate_leaves(j2)

        return list(chain(*map(generate_leaves, self.coupling.items())))


    @property
    def states(self):
        State = namedtuple('state', self.basis)
        func = {
            '2': lambda o: (self.qn[o],),
            'z': lambda o: np.linspace(-self.qn[o], self.qn[o], self.mult[o])
        }
        return [State(*qns) for qns in product(
            *(func[comp](op) for op, comp in self.basis))]

    def get_op(self, op):

        for j, j1j2 in self.coupling.items():
            if j == op:
                j_op = jnp.array(calc_ang_mom_ops(self.qn[op])[0:3])
                return reduce(jnp.kron, [j_op if o == op else jnp.identity(m)
                                         for o, m in self.mult.items()])
            elif j1j2 is not None:
                try:
                    sym = Symbol(coupling={k: j1j2 for k, j1j2 in self.coupling.items() if k != j} | dict(j1j2),
                                 **{k: qn for k, qn in self.qn.items() if k != j})
                    j_op = sym.get_op(op)
                    (j1, _), (j2, _) = j1j2
                    _, cg_vec = sym.couple(j, j1, j2, levels=[self])
                    return unitary_transform(j_op, cg_vec)

                except ValueError:
                    pass
        else:
            raise ValueError(f"Angular momentum label {op} is undefined!")

    def couple(self, j, j1, j2, levels=None):

        levels = levels or self.levels(j, j1, j2)

        uncpld_basis = [j + comp for j, comp in product((j1, j2), ('2', 'z'))]
        cpld_basis = [j + comp for comp in ('2', 'z')]
        extra_basis = [op for op in self.basis if op not in uncpld_basis]

        def qns(uncpld_state, cpld_state):
            j1m1j2m2 = tuple(getattr(uncpld_state, op) for op in uncpld_basis)
            jm = tuple(getattr(cpld_state, op) for op in cpld_basis)
            return j1m1j2m2 + jm

        def equal_extra(uncpld_state, cpld_state):
            for op in extra_basis:
                yield getattr(uncpld_state, op) == getattr(cpld_state, op)

        cg_vec = jnp.array(
            [[float(CG(*qns(uncpld, cpld)).doit())
              if all(equal_extra(uncpld, cpld)) else 0.0
              for lvl in levels for cpld in lvl.states]
             for uncpld in self.states])

        return levels, cg_vec

    def levels(self, j, j1, j2, cls=None):

        qn1, qn2 = self.qn[j1], self.qn[j2]

        cplg = {o: c for o, c in self.coupling.items() if o not in (j1, j2)} \
            | {j: ((j1, self.coupling[j1]), (j2, self.coupling[j2]))}

        uncpld_qn = {op: qn for op, qn in self.qn.items() if op not in cplg}
        cpld_qn = [{op: qn if op == j else self.qn[op] for op in cplg}
                   for qn in np.arange(np.abs(qn1 - qn2), qn1 + qn2 + 1)]

        if cls is None:
            cls = Symbol

        return [cls(coupling=cplg, **uncpld_qn, **cpld) for cpld in cpld_qn]

    def rotate(self, space=None, quax=None):

        if quax is not None:
            rot = R.from_matrix(quax.T)

        totj = np.sum([self.get_op(o) for o in space or self.coupling], axis=0)
        rotation = expm(-1.j * np.einsum('i,ijk->jk', rot.as_rotvec(), totj))

        return rotation

    def reduce(self, ops=None, cls=None):

        if cls is None:
            cls = Symbol

        red = cls(**{op: self.qn[op] for op in ops or self.coupling})
        perm = [red.states.index(tuple(getattr(state, op) for op in red.basis))
                for state in self.states]

        return red, perm

    def __contains__(self, other):
        return all(self.qn[key] == val for key, val in other.qn.items())

    def __str__(self):
        qn_str = ', '.join(f"{op}={qn}" for op, qn in self.qn.items())
        return f'{self.__class__.__name__}({qn_str})'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.coupling},**{self.qn})'


def couple(space, **coupling):

    if coupling:

        j, (j1, j2) = coupling.popitem()
        lvls, cg_vec = couple(space, **coupling)

        level_blks, cg_vec_blks = zip(*[lvl.couple(j, j1, j2) for lvl in lvls])

        return ([lvl for blk in level_blks for lvl in blk],
                cg_vec @ block_diag(*cg_vec_blks))

    else:
        return [space], jnp.identity(space.multiplicity)


spec2angm = {
    'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'K': 7,
    'L': 8, 'M': 9, 'N': 10, 'O': 11, 'Q': 12, 'R': 13
}

angm2spec = {
    0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I', 7: 'K',
    8: 'L', 9: 'M', 10: 'N', 11: 'O', 12: 'Q', 13: 'R'
}


class Term(Symbol):

    def __init__(self, coupling=None, **qn):
        super().__init__(coupling=coupling, **qn)
        self.spin_mult = 2 * self.qn['S'] + 1
        self.orb_letter = angm2spec[self.qn['L']]

    def levels(self, j='J', j1='L', j2='S'):
        return super().levels(j, j1, j2, cls=Level)

    @classmethod
    def parse(cls, symbol_str, spin_major=True):
        m = re.match(r'(?P<S>\d+)(?P<L>[A-Z])$', symbol_str)

        if m:
            d = m.groupdict()
        else:
            raise ValueError("Expected form like 6H.")

        if spin_major:
            return cls(L=spec2angm[d['L']], S=(int(d['S']) - 1) / 2)
        else:
            return cls(S=(int(d['S']) - 1) / 2, L=spec2angm[d['L']])

    def __str__(self):
        return ''.join(map(str, [self.mult['S'], self.orb_letter]))

    def __repr__(self):
        return f'Term({self.mult["S"]},{self.orb_letter})'


class Level(Symbol):

    def __init__(self, coupling=None, **qn):

        if 'J' not in coupling:
            raise ValueError("Level needs to include J-coupling!")

        super().__init__(coupling=coupling, **qn)
        self.spin_mult = 2 * self.qn['S'] + 1
        self.orb_letter = angm2spec[self.qn['L']]

    @classmethod
    def parse(cls, symbol_str, spin_major=True):
        m = re.match(r'(?P<S>\d+)(?P<L>[A-Z])(?P<Jn>\d+)(?:\/(?P<Jd>\d+))?$',
                     symbol_str)
        if m:
            d = m.groupdict()
        else:
            raise ValueError("Expected form like 6H15/2.")

        if spin_major:
            return cls(coupling={'J': (('L', None), ('S', None))},
                       S=(int(d['S']) - 1) / 2, L=spec2angm[d['L']],
                       J=int(d['Jn']) / (int(d['Jd'] or 1)))
        else:
            return cls(coupling={'J': (('S', None), ('L', None))},
                       L=spec2angm[d['L']], S=(int(d['S']) - 1) / 2,
                       J=int(d['Jn']) / (int(d['Jd'] or 1)))

    def __str__(self):
        return ''.join(map(str, [
            int(2 * self.qn['S']) + 1,
            self.orb_letter,
            Fraction(self.qn['J'])]))

    def __repr__(self):
        print(self.mult)
        return f'Level({self.spin_mult},{self.orb_letter},{self.qn["J"]})'


def parse_termsymbol(symbol_str):
    try:
        return Level.parse(symbol_str)
    except ValueError:
        try:
            return Term.parse(symbol_str)
        except ValueError:
            return eval(f"Symbol({symbol_str})", {"Symbol": Symbol})
