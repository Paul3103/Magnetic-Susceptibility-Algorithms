
"""
This module contains functions for XYZABSCDEF
"""

import re
import numpy as np
import numpy.linalg as la

from .basis import Term, Level


def calc_lande_g(J, L, S):
    """
    Calculates lande g factor

    Parameters
    ----------
    J : float
        J quantum number
    L : float
        L quantum number
    S : float
        S quantum number

    Returns
    -------
    float
        Lande g factor

    """

    # Orbital part
    gJ = (J*(J+1.) - S*(S+1.) + L*(L+1.))/(2.*J*(J+1.))

    # Spin part
    gJ += (J*(J+1.) + S*(S+1.) - L*(L+1.))/(J*(J+1.))

    return gJ


def calc_mag_moment_j(J, L, S, jx, jy, jz):
    """
    Calculate magnetic moment in total angular momentum basis |J, mJ>
    along each axis x, y, and z with units [cm-1 T-1]

    Parameters
    ----------
    J : float
        J quantum number
    L : float
        L quantum number
    S : float
        S quantum number
    jx : np.ndarray
        Matrix representation of x angular momentum operator
    jy : np.ndarray
        Matrix representation of y angular momentum operator
    jz : np.ndarray
        Matrix representation of z angular momentum operator

    Returns
    -------
    np.ndarray
        Magnetic moment operator in x direction [cm-1 T-1]
    np.ndarray
        Magnetic moment operator in y direction [cm-1 T-1]
    np.ndarray
        Magnetic moment operator in z direction [cm-1 T-1]

    """

    _jx = np.complex128(jx)
    _jy = np.complex128(jy)
    _jz = np.complex128(jz)

    gJ = calc_lande_g(J, L, S)

    mu_B = 0.466866577042538

    mu_x = gJ * _jx * mu_B
    mu_y = gJ * _jy * mu_B
    mu_z = gJ * _jz * mu_B

    return mu_x, mu_y, mu_z


def calc_HZee_j(J, L, S, jx, jy, jz, B):
    """
    Calculate Zeeman Hamiltonian in total angular momentum basis |J, mJ>

    Parameters
    ----------
    J : float
        J quantum number
    L : float
        L quantum number
    S : float
        S quantum number
    jx : np.ndarray
        Matrix representation of x angular momentum operator
    jy : np.ndarray
        Matrix representation of y angular momentum operator
    jz : np.ndarray
        Matrix representation of z angular momentum operator
    B : np.ndarray
        Magnetic field strengths in x, y, z - [Bx, By, Bz] in Tesla

    Returns
    -------
    np.ndarray
        Magnetic moment operator in x direction
    np.ndarray
        Magnetic moment operator in y direction
    np.ndarray
        Magnetic moment operator in z direction

    """

    # Make sure operators are complex type
    _jx = np.complex128(jx)
    _jy = np.complex128(jy)
    _jz = np.complex128(jz)

    # Magnetic moment with units [cm-1 T-1]
    _mu_x, _mu_y, _mu_z = calc_mag_moment_j(J, L, S, _jx, _jy, _jz)

    # Form Zeeman Hamiltonian
    HZee = _mu_x * B[0] + _mu_y * B[1] + _mu_z * B[2]

    # Diagonalise
    Zee_val, Zee_vec = la.eigh(HZee)

    # Set ground energy to zero
    Zee_val -= Zee_val[0]

    return HZee, Zee_val, Zee_vec


def hunds_ground_term(n_elec, n_orb):
    """
    Calculate J, L, and S quantum numbers using number of
    electrons and orbitals

    Parameters
    ----------
    n_elec : int
        Number of electrons
    n_orb  : int, {3, 5, 7}
        Number of orbitals

    Returns
    -------
    float
        Total angular momentum quantum number J
    float
        Orbital angular momentum quantum number L
    float
        Spin angular momentum quantum number S
    """

    # Set constants for given shell
    if n_orb == 7:
        ml_vals = [3, 2, 1, 0, -1, -2, -3]
        max_s = 3.5
    elif n_orb == 5:
        ml_vals = [2, 1, 0, -1, -2]
        max_s = 2.5
    elif n_orb == 3:
        ml_vals = [1, 0, -1]
        max_s = 1.5
    else:
        print('Unsupported number of orbitals: {:d}'.format(n_orb))
        exit()

    # More than half filled
    if n_elec > n_orb:
        s = max_s - float(n_elec - n_orb) * 0.5
        lqn = float(sum(ml_vals[:n_elec - n_orb]))
        j = lqn + s
    # Less than half filled
    elif n_elec < n_orb:
        s = 0.5 * float(n_elec)
        lqn = float(sum(ml_vals[:n_elec]))
        j = lqn - s
    # Half filled
    elif n_elec == n_orb:
        s = max_s
        lqn = 0.
        j = 0.

    return j, lqn, s


class Ion:

    def __init__(self, chrg, elem=None, **shells):
        self.chrg = chrg
        self.elem = elem
        self.shells = shells

    @property
    def levels(self):
        levels = {
            ('Dy', 3): (Level(6, 'H', 15/2), Level(6, 'H', 13/2),
                        Level(6, 'H', 11/2), Level(6, 'H', 9/2),
                        Level(6, 'F', 11/2), Level(6, 'H', 7/2),
                        Level(6, 'F', 9/2), Level(6, 'H', 5/2),
                        Level(6, 'F', 7/2), Level(6, 'F', 5/2),
                        Level(6, 'F', 3/2), Level(6, 'F', 1/2))
        }[(self.elem, self.chrg)]
        return levels

    @property
    def ground_level(self):

        levels = {
            ('Dy', 3): Level(6, 'H', 15/2)
        }

        return levels[(self.elem, self.chrg)]

    @property
    def ground_term(self):

        terms = {
            ('Dy', 3): Term(6, 'H')
        }

        return terms[(self.elem, self.chrg)]

    def casscf_roots(self, size):

        nroots = {}

        for sym in self.casscf_levels(size):
            try:
                nroots[sym.spin_mult] += sym.mult / sym.spin_mult
            except KeyError:
                nroots[sym.spin_mult] = sym.mult / sym.spin_mult

        return nroots

    def casscf_terms(self, size):
        terms = {
            ('Dy', 3): {'s': [Term(6, 'H'), Term(6, 'F')]}
        }
        return terms[(self.elem, self.chrg)][size]

    def casscf_levels(self, size):
        return [lvl for trm in self.casscf_terms(size) for lvl in trm.levels()]

    def theta(self, basis):

        # taken from the phi manual
        theta_j = {
            ('Ce', 3): {2: -2/35, 4: 2/315, 6: 0.0},
            ('Pr', 3): {2: -52/2475, 4: -4/5445, 6: 272/4459455},
            ('Nd', 3): {2: -7/1089, 4: -136/467181, 6: -1615/42513471},
            ('Pm', 3): {2: 14/1815, 4: 952/2335905, 6: 2584/42513471},
            ('Sm', 3): {2: 13/315, 4: 26/10395, 6: 0.0},
            ('Eu', 3): {2: 0.0, 4: 0.0, 6: 0.0},
            ('Gd', 3): {2: 0.0, 4: 0.0, 6: 0.0},
            ('Tb', 3): {2: -1/99, 4: 2/16335, 6: -1/891891},
            ('Dy', 3): {2: -2/315, 4: -8/135135, 6: 4/3864861},
            ('Ho', 3): {2: -1/450, 4: -1/30030, 6: -5/3864861},
            ('Er', 3): {2: 4/1575, 4: 2/45045, 6: 8/3864861},
            ('Tm', 3): {2: 1/99, 4: 8/49005, 6: -5/891891},
            ('Yb', 3): {2: 2/63, 4: -2/1155, 6: 4/27027}
        }

        theta_l = {
            ('Ce', 3): {2: -2/45, 4: 2/495, 6: -4/3861},
            ('Pr', 3): {2: -2/135, 4: -4/10395, 6: 2/81081},
            ('Nd', 3): {2: -2/495, 4: -2/16335, 6: -10/891891},
            ('Pm', 3): {2: 2/495, 4: 2/16335, 6: 10/891891},
            ('Sm', 3): {2: 2/135, 4: 4/10395, 6: -2/81081},
            ('Eu', 3): {2: 2/45, 4: -2/495, 6: 4/3861},
            ('Gd', 3): {2: 0.0, 4: 0.0, 6: 0.0},
            ('Tb', 3): {2: -2/45, 4: 2/495, 6: -4/3861},
            ('Dy', 3): {2: -2/135, 4: -4/10395, 6: 2/81081},
            ('Ho', 3): {2: -2/495, 4: -2/16335, 6: -10/891891},
            ('Er', 3): {2: 2/495, 4: 2/16335, 6: 10/891891},
            ('Tm', 3): {2: 2/135, 4: 4/10395, 6: -2/81081},
            ('Yb', 3): {2: 2/45, 4: -2/495, 6: 4/3861}
        }

        if basis == 'j':
            return theta_j[(self.elem, self.chrg)]
        elif basis == 'l':
            return theta_l[(self.elem, self.chrg)]
        else:
            raise ValueError("Unknown basis <{}>".format(basis))

    @classmethod
    def from_elem_chrg(cls, elem, chrg):

        # todo: check
        shells = {
            ('Ce', 3): {'f': 1},
            ('Pr', 3): {'f': 2},
            ('Nd', 3): {'f': 3},
            ('Pm', 3): {'f': 4},
            ('Sm', 3): {'f': 5},
            ('Eu', 3): {'f': 6},
            ('Gd', 3): {'f': 7},
            ('Tb', 3): {'f': 8},
            ('Dy', 3): {'f': 9},
            ('Ho', 3): {'f': 10},
            ('Er', 3): {'f': 11},
            ('Tm', 3): {'f': 12},
            ('Yb', 3): {'f': 13}
        }

        return cls(chrg, elem=elem, **shells[(elem, chrg)])

    @classmethod
    def parse(cls, label):
        m = re.match(r'(?P<elem>\w+)(?P<chrg>\d+)(?P<sgn>[+-])', label)
        m_dict = m.groupdict()
        if m_dict['sgn'] == '+':
            chrg = int(m_dict['chrg'])
        elif m_dict['sgn'] == '-':
            chrg = -int(m_dict['chrg'])
        else:
            raise ValueError('Error while parsing sign.')

        return cls.from_elem_chrg(m_dict['elem'], chrg)

    def __str__(self):
        return "{}{:d}{}".format(self.elem, abs(self.chrg),
                                 '+' if self.chrg > 0 else '-')

    def __repr__(self):
        return f'Ion({self.chrg}, elem="{self.elem}")'

    def __hash__(self):
        return hash(self.__repr__())
