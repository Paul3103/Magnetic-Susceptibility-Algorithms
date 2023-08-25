#! /usr/bin/env python3

"""
This module contains various utility functions
"""

from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt


def krodelta(arg1, arg2):
    """
    Kronecker delta function

    Parameters
    ----------
        arg1 : float or int
            First variable
        arg2 : float or int
            Second variable

    Returns
    -------
        int
            1 if arg1 == arg, else 0
    """

    if arg1 == arg2:
        val = 1
    else:
        val = 0

    return val


@lru_cache(maxsize=None)
def GCD(a, b):
    """
    Calculates greatest common divisor of two numbers
    by recursion

    Parameters
    ----------
        a : float or int
            first number
        b : float or int
            second number

    Returns
    -------
        float
            greatest common divisor of a and b

    """
    if b == 0:
        divisor = a
    else:
        divisor = GCD(b, a % b)

    return divisor


def binomial(n, r):
    """
    Calculates binomial coefficient nCr (n r)

    Parameters
    ----------
    n : int or float
        first number
    n : int or float
        second number

    Returns
    -------
    bcoeff : float
        Binomial coefficient

    """
    bcoeff = 0.

    if r == n or r == 0:
        bcoeff = 1.
    elif r < 0:
        bcoeff = 0.
    elif r == 1:
        bcoeff = n
    else:
        #https://en.wikipedia.org/wiki/Binomial_coefficient#Multiplicative_formula # noqa
        bcoeff = n
        for i in range(2, r+1):
            bcoeff *= (n + 1 - i) / i

    return bcoeff


def hex_to_rgb(value):
    """
    Convert hex code to rgb list

    Parameters
    ----------
    value : str
        Hex code

    Returns
    -------
    list
        [red, green, blue]
    """

    value = value.lstrip('# ')
    lv = len(value)
    rgb = [int(value[i:i + lv // 3], 16)/255. for i in range(0, lv, lv // 3)]

    return rgb


def plot_op(ops, f_plot, titles=None, sq=False):
    """Plot list of complex operators to pdf.

    Parameters
    ----------
    ops : list of np.arrays, dtype=complex
        List of operators in matrix form.
    f_plot : str
        pdf output file name.
    titles : list
        List of plot titles, same length as ops.
    sq : bool
        Flag to control printing of the squared operator.

    Returns
    -------
    None
    """
    fig = plt.figure()

    num = len(ops)
    tot_num = num + (1 if sq else 0)

    fig.set_size_inches(8, tot_num * 4)

    for i, op in enumerate(ops):
        name = ("component-{}".format(i + 1)) if titles is None else titles[i]

        ax_r = fig.add_subplot(tot_num, 2, 2 * i + 1)
        plt.title("{} (real)".format(name))
        real = ax_r.matshow(np.real(op))
        fig.colorbar(real)

        ax_i = fig.add_subplot(tot_num, 2, 2 * i + 2)
        plt.title("{} (imag)".format(name))
        imag = ax_i.matshow(np.imag(op))
        fig.colorbar(imag)

    if sq:
        sq = np.sum([ops[comp] @ ops[comp] for comp in range(3)], axis=0)

        ax_sq_r = fig.add_subplot(tot_num, 2, 2 * num + 1)
        plt.title("squared (real)")
        sq_real = ax_sq_r.matshow(np.real(sq))
        fig.colorbar(sq_real)

        ax_sq_i = fig.add_subplot(tot_num, 2, 2 * num + 2)
        plt.title("squared (imag)")
        sq_imag = ax_sq_i.matshow(np.imag(sq))
        fig.colorbar(sq_imag)

    plt.savefig(f_plot, dpi=600)
