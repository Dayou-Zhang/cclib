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

    def _set_from_attr(self, source_name, dest_name, scalar=False):
        value = self.h5.attrs.get(source_name, None)
        if value is not None:
            if scalar:
                value = value.item()
            setattr(self.data, dest_name, value)

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
        self._meta_from_attr('NSYM', 'nsym', True)
        print(data.metadata)
        self._set_from_attr('NBAS', 'nbasis', True)
        self._set_from_attr('SPINMULT', 'mult', True)
        self._set_from_dataset('AO_OVERLAP_MATRIX', 'aooverlaps', (data.nbasis, data.nbasis))
        self._set_from_dataset('BASIS_FUNCTION_IDS', 'aoqnums')
        data.aoqnums[:, 0] -= 1
        self._set_from_dataset('CENTER_ATNUMS', 'atomnos')
        data.natom = len(data.atomnos)
        self._set_from_dataset('CENTER_COORDINATES', 'atomcoords')
        self._set_from_dataset('MO_ENERGIES', 'moenergies')
        self.moenergies = [data.moenergies[0]]
        self._set_from_dataset('MO_OCCUPATIONS', 'nooccnos')
        self._set_from_dataset('MO_VECTORS', 'nocoeffs', (data.nbasis, data.nbasis))
        self.mocoeffs = [data.nocoeffs]
        self._set_from_dataset('ROOT_ENERGIES', 'scfenergies')

        del self.data
        return ccData(vars(data))

