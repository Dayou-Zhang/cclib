# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""Parser for Molcas H5 files"""

from types import SimpleNamespace

import numpy

from cclib.parser import logfileparser
from cclib.parser import utils
from cclib.parser.data import ccData

SHELL_LABELS = "SPDFGHI"

class MolcasH5:
    """A Molcas H5 file."""

    def __init__(self, h5, *args, **kwargs):
        self.h5 = h5

    def __str__(self):
        """Return a string repeesentation of the object."""
        return f"Molcas H5 file {self.filename}"

    def __repr__(self):
        """Return a representation of the object."""
        return f'MolcasH5("{self.filename}")'

    def normalisesym(self, label):
        """Normalise the symmetries used by Molcas.

        The labels are standardized except for the first character being lowercase.
        """
        return label[0].upper() + label[1:]

    def _meta_from_attr(self, source_name, dest_name, scalar=False):
        value = self.h5.attrs.get(source_name, None)
        if value is not None:
            if scalar:
                value = value.item()
            self.data.metadata[dest_name] = value
            return value

    def _set_from_attr(self, source_name, dest_name, scalar=False):
        value = self.h5.attrs.get(source_name, None)
        if value is not None:
            if scalar:
                value = value.item()
            setattr(self.data, dest_name, value)
            return value

    def _set_from_dataset(self, source_name, dest_name, reshape=None):
        """reshape: tuple - reshape as shape
                    True - treat as scalar
                    other - do not reshape"""
        value = self.h5.get(source_name, None)
        if value is not None:
            value = value[()]
            if reshape is True:
                value = value.item()
            elif isinstance(reshape, tuple):
                value = value.reshape(reshape)
            setattr(self.data, dest_name, value)
            return value

    def unpack(self, vector, nbas, nbasis):
        """When symmetry is used, 2D matrix is stored as multiple
        smaller square matrices. This method unpacks the condensed
        form into a single (nbasis, nbasis) matrix."""
        if len(nbas) == 1:
            return vector.reshape((nbasis, nbasis))
        # idxb? is the basis index in the square matrix
        idxb0 = 0
        # idxv? is the vector index
        idxv0 = 0
        matrix = numpy.zeros((nbasis, nbasis))
        for l in nbas:
            idxb1 = idxb0 + l
            idxv1 = idxv0 + l * l
            matrix[idxb0:idxb1, idxb0:idxb1] = vector[idxv0:idxv1].reshape((l, l))
            idxb0 = idxb1
            idxv0 = idxv1
        assert idxv1 == len(vector), "Wrong size while unpacking"
        return matrix

    def parse(self):
        h5 = self.h5
        data = SimpleNamespace()
        self.data = data
        data.metadata = {
            'methods': [h5.attrs['MOLCAS_MODULE'].decode('utf-8')],
            'package': 'Molcas',
            'package_version': h5.attrs['MOLCAS_VERSION'].decode('utf-8'),
        }
        self._meta_from_attr('NACTEL', 'nactel', True)
        self._meta_from_attr('NROOTS', 'nroots', True)
        self._meta_from_attr('NSTATES', 'nstates', True)
        nsym = self._meta_from_attr('NSYM', 'nsym', True)
        nbas = self._meta_from_attr('NBAS', 'nbas')
        nbasis = data.nbasis = nbas.sum().item()
        self._set_from_dataset('AO_OVERLAP_MATRIX', 'aooverlaps')
        data.aooverlaps = self.unpack(data.aooverlaps, nbas, nbasis)
        self._set_from_dataset('BASIS_FUNCTION_IDS', 'aoqnums')
        self._set_from_dataset('DESYM_BASIS_FUNCTION_IDS', 'aoqnums')
        data.aoqnums[:, 0] -= 1
        self._set_from_dataset('CENTER_ATNUMS', 'atomnos')
        self._set_from_dataset('DESYM_CENTER_ATNUMS', 'atomnos')
        data.natom = len(data.atomnos)
        mult = self._set_from_attr('SPINMULT', 'mult', True)
        if mult is None:
            mult = data.mult = data.atomnos.sum().item() % 2 + 1
        self._set_from_dataset('CENTER_COORDINATES', 'atomcoords')
        self._set_from_dataset('DESYM_CENTER_COORDINATES', 'atomcoords')
        data.atomcoords = [data.atomcoords]
        self._set_from_dataset('MO_ENERGIES', 'moenergies')
        data.moenergies = [utils.convertor(data.moenergies, 'hartree', 'eV')]
        self._set_from_dataset('MO_OCCUPATIONS', 'nooccnos')
        self._set_from_dataset('MO_VECTORS', 'nocoeffs')
        data.nocoeffs = self.unpack(data.nocoeffs, nbas, nbasis)
        data.mocoeffs = [data.nocoeffs]
        self._set_from_dataset('ENERGY', 'scfenergies', (1,))
        self._set_from_dataset('ROOT_ENERGIES', 'scfenergies')
        data.scfenergies = utils.convertor(data.scfenergies, 'hartree', 'eV')
        data.charge = round(data.atomnos.sum() - data.nooccnos.sum())
        nelec = round(data.nooccnos.sum())
        homoa = (nelec + mult - 3) // 2
        homob = homoa + 1 - mult
        data.homos = (homoa, homob)

        primitives = h5['PRIMITIVES'][()]
        primitive_ids = h5['PRIMITIVE_IDS'][()]
        gbasis = []
        old_atom = None
        old_l_qn = None
        old_contraction_id = None
        for (exponent, coeff), (atom, l_qn, contraction_id) in zip(primitives, primitive_ids):
            if old_atom != atom:
                current_atom = []
                gbasis.append(current_atom)
                old_atom = atom
                old_l_qn = None
                old_contraction_id = None
            if old_l_qn != l_qn or old_contraction_id != contraction_id:
                current_contraction = []
                current_atom.append((SHELL_LABELS[l_qn], current_contraction))
                old_l_qn = l_qn
                old_contraction_id = contraction_id
            if coeff != 0.:
                current_contraction.append((exponent, coeff))
        data.gbasis = gbasis

        del self.data
        return ccData(vars(data))

