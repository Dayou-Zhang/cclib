import numpy
import copy

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

def _reorder_array(arr, order, dim, key):
    ndim = arr.ndim
    assert len(dim) == ndim, f'length of dim does not match in {key}: should be {ndim}, got {len(dim)}'
    idx_empty = [slice(None)] * ndim
    for n, b in enumerate(dim):
        if not b:
            continue
        idx = idx_empty.copy()
        idx[n] = order
        arr[...] = arr[tuple(idx)]
    return arr

def _reorder_by_attr(data, key, order, dim, crash_if_not_present=False, print_warning=True):
    """Reorder the array of `data.key` by int array `order` which contains
    index to transform to the new order. Arrays are modified in place.
    `dim` is a bool tuple indicating which dimension of the
    array should be converted.
    If crash_if_not_present is True, the method would raise an error
    Returns the new array if converted, otherwise return None"""
    arr = getattr(data, key, None)
    if arr is None:
        if crash_if_not_present:
            raise KeyError(f"Missing {key} in {data}")
        else:
            if print_warning:
                print(f"Warning: Missing {key} in {data}")
            return
    if isinstance(arr, list):
        arr = numpy.asarray(arr)
        _reorder_array(arr, order, dim, key)
        setattr(data, key, arr)
        return arr
    else:
        assert isinstance(arr, numpy.ndarray)
        return _reorder_array(arr, order, dim, key)

def reorder_ao_by_qnums(data, ref_aoqnums):
    data = copy.deepcopy(data)
    assert data.nbasis == len(ref_aoqnums)
    order = map_a_to_b(data.aoqnums, ref_aoqnums)
    _reorder_by_attr(data, 'mocoeffs', order, (False, True, False))
    _reorder_by_attr(data, 'moenergies', order, (False, True))
    return data

