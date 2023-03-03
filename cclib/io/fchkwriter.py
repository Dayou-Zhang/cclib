# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""A writer for Gaussian FChk files."""

import numpy

from cclib.io import filewriter
from cclib.parser import utils

SHELL_LABELS = "SPDFGHI"

def write_scalar_record(lines, name, value):
    if isinstance(value, (int, numpy.integer)):
        lines.append(f'{name:40s}   I     {value:12d}\n')
    elif isinstance(value, float):
        lines.append(f'{name:40s}   R     {value:22.15E}\n')
    elif isinstance(value, str):
        str_l = len(value)
        rec_l, remainder = divmod(str_l, 12)
        if remainder > 0:
            rec_l += 1
            value += ' ' * (12 - remainder)
        lines.append(f'{name:40s}   C   N={rec_l:12d}\n')
        while len(value) > 0:
            if len(value) > 60:
                line = value[:60]
                value = value[60:]
            else:
                line = value
                value = ""
            lines.append(line + '\n')
    else:
        raise ValueError(f"Unsupported type {type(value)}")

def write_array_record(lines, name, value):
    value = numpy.asarray(value)
    if numpy.issubdtype(value.dtype, numpy.floating):
        lines.append(f'{name:40s}   R   N={value.size:12d}\n')
        data = numpy.array2string(value, max_line_width=81, separator=' ',
                formatter={'all':lambda x: '%15.8E' % x})[1:-1]
        lines.append(f' {data}\n')
    elif numpy.issubdtype(value.dtype, numpy.integer):
        lines.append(f'{name:40s}   I   N={value.size:12d}\n')
        data = numpy.array2string(value, max_line_width=73, separator=' ',
                formatter={'all':lambda x: '%11d' % x})[1:-1]
        lines.append(f' {data}\n')
    else:
        raise ValueError(f"Unsupported dtype {value.dtype}")


class FChk(filewriter.Writer):
    """A writer for Gaussian FChk files."""

    def __init__(self, ccdata, title=None, jobtype=None, method_name=None, basis_name=None, *args, **kwargs):
        """Initialize the FChk writer object.

        Inputs:
          ccdata - An instance of ccData, parse from a logfile.
        """
        super().__init__(ccdata, *args, **kwargs)

        self.title = title
        self.jobtype = jobtype
        self.method_name = method_name
        self.basis_name = basis_name

        self.required_attrs = ('natom', 'charge', 'mult', 'homos', 'nbasis', 'atomnos', 'atomcoords', 'aoqnums', 'gbasis', 'mocoeffs', 'moenergies')

    def generate_repr(self):
        data = self.ccdata
        f = []
        with numpy.printoptions(threshold=numpy.inf):
            try:
                metadata = data.metadata
                if self.method_name is None:
                    method_name = metadata.get('methods', [])
                    try:
                        method_name = method_name[-1]
                    except IndexError:
                        method_name = None
                if self.basis_name is None:
                    basis_name = metadata.get('basis_set', None)
            except AttributeError:
                pass
            title = self.title
            if title is None:
                title = 'Untitled'
            jobtype = self.jobtype
            if jobtype is None:
                jobtype = 'SP'
            if method_name is None:
                method_name = 'Unknown'
            if basis_name is None:
                basis_name = 'Gen'
            f.append(f'{title:72s}\n{jobtype:10s}{method_name:30s}{basis_name:30s}\n')
            write_scalar_record(f, 'Number of atoms', data.natom)
            write_scalar_record(f, 'Charge', data.charge)
            write_scalar_record(f, 'Multiplicity', data.mult)
            neleca = data.homos
            try:
                neleca, nelecb = neleca
            except ValueError:
                neleca = nelecb = neleca.item()
            neleca += 1
            nelecb += 1
            write_scalar_record(f, 'Number of electrons', neleca + nelecb)
            write_scalar_record(f, 'Number of alpha electrons', neleca)
            write_scalar_record(f, 'Number of beta electrons', nelecb)
            write_scalar_record(f, 'Number of basis functions', data.nbasis)
            write_array_record(f, 'Atomic numbers', data.atomnos)
            write_array_record(f, 'Nuclear charges', data.atomnos.astype(float))
            atomcoords = utils.convertor(data.atomcoords, 'Angstrom', 'bohr').ravel()
            write_array_record(f, 'Current cartesian coordinates', atomcoords)
            shell_types = []
            nprimitives = []
            shell_map = []
            exponents = []
            contraction = []
            aoqnums = iter(data.aoqnums)
            for atom, atom_basis in enumerate(data.gbasis, 1):
                for shell, primitives in atom_basis:
                    l = SHELL_LABELS.index(shell)
                    q = next(aoqnums)
                    n = len(primitives)
                    if q[-1] < 10:
                        # spherical basis
                        for _ in range(2 * l):
                            next(aoqnums)
                        if l > 1:
                            l = -l
                    else:
                        # cartesian basis
                        for _ in range((l + 1) * (l + 2) // 2 - 1):
                            next(aoqnums)
                    shell_types.append(l)
                    nprimitives.append(n)
                    shell_map.append(atom)
                    for e, c in primitives:
                        exponents.append(e)
                        contraction.append(c)
            write_scalar_record(f, 'Number of contracted shells', len(shell_types))
            write_scalar_record(f, 'Number of primitive shells', len(exponents))
            write_array_record(f, 'Shell types', shell_types)
            write_array_record(f, 'Number of primitives per shell', nprimitives)
            write_array_record(f, 'Shell to atom map', shell_map)
            write_array_record(f, 'Primitive exponents', exponents)
            write_array_record(f, 'Contraction coefficients', contraction)
            mo_energy = [utils.convertor(i, 'eV', 'hartree') for i in data.moenergies]
            mo_coeff = data.mocoeffs
            write_array_record(f, 'Alpha Orbital Energies', mo_energy[0].ravel())
            if len(mo_energy) > 1:
                write_array_record(f, 'Beta Orbital Energies', mo_energy[1].ravel())
            write_array_record(f, 'Alpha MO coefficients', mo_coeff[0].ravel())
            if len(mo_coeff) > 1:
                write_array_record(f, 'Beta MO coefficients', mo_coeff[1].ravel())
        return ''.join(f)

