import numpy as np
from functools import reduce


def project_angm(j, j2, jmax=None):

    n = j2.shape[1]

    if (j * 2) % 2 == 0:
        jmax = jmax or 100
        irreps = np.arange(jmax + 1)
    elif (j * 2) % 2 == 1:
        jmax = jmax or 100.5
        irreps = np.arange(1/2, jmax + 1)
    else:
        raise ValueError(f"j = {j} must be an exact (half-)integer number!")

    def jval(k):
        return k * (k + 1)

    def terms():
        for l in irreps:
            if l == j:
                continue

            yield (j2 - jval(l) * np.identity(n)) / (jval(j) - jval(l))

    return reduce(np.matmul, terms())


def cartesian_to_polar_basis(j_ops):
    j_p = (j_ops[0] + 1.j * j_ops[1]) / np.sqrt(2)
    j_m = (j_ops[0] - 1.j * j_ops[1]) / np.sqrt(2)
    return [j_ops[2]], [[j_p, j_m]]


def project_semi_simple_lie_group(group, weight, irrep, h_ops, e_ops, roots):
    pass

def project_simple_compact_lie_group(group, weight, irrep, h_ops, e_ops, roots):

    if group == "SO(3)":
        pos_roots = [np.array([1.0])]

    rho = sum(pos_roots)



def project_simple_lie_group(group, weight, irrep, h_ops, e_ops, roots):
    """
    group : Group symbol in Cartan classification.
    weight : Weight vector.
    irrep : Heighest weight vector of irrep.
    h_ops : List of infinitesimal diagonal operators.
    e_ops : List of infinitesimal ladder operators.
    """

    # check that irrep corresponds to heighest root of some irrep

    if weight != irrep:  # arbitrary weight
        # projection operator of heighest weight
        proj_max = project_simple_lie_group(group, irrep, irrep, e_ops, h_ops)
        return lowering(weight) @ proj_max @ raising(weight)



