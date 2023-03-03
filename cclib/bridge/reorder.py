import numpy

def map_a_to_b(a, b):
    """Find the index idx such that a[idx] gives the same row order as b
    This method assumes doing a lexsort from the 1st to last column
    of a and b gives the same logical order.
    This method is useful for converting AO indices between different
    softwares, so that one can port MO coefficients etc from one
    software to another.
    a and b are both array like of shape (m, n)
    return value idx has shape (m,)"""
    a_rev = tuple(reversed(numpy.asarray(a).T))
    b_rev = tuple(reversed(numpy.asarray(b).T))
    return numpy.lexsort(a_rev)[numpy.argsort(numpy.lexsort(b_rev))]

