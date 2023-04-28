# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""Parser for Molcas output files"""

import re
import string

import numpy
import itertools
from collections import OrderedDict

from cclib.parser import logfileparser
from cclib.parser import utils

SHELL_ML_SPHERICAL = [
    # 0: S
    [0],
    # 1: P
    [1, -1, 0],
    # 2: D
    [0, 1, -1, 2, -2],
    # 3: F
    [0, 1, -1, 2, -2, 3, -3],
    # 4: G
    [0, 1, -1, 2, -2, 3, -3, 4, -4],
]

SHELL_ML_CARTESIAN = [
    # 0: S
    [0],
    # 1: P
    [1, -1, 0],
    # Treat XYZ as two binary digits: X -> 0b01, Y -> 0b10, Z -> 0b11
    # To make cartesian easy to distinguish from spherical, all values
    #    are <<ed by 1
    # e.g. XYZ -> 0b0110110 -> 54
    # 2: D
    # ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']
    [10, 20, 30, 12, 14, 22],
    # 3: F
    # ['XXX', 'YYY', 'ZZZ', 'XYY', 'XXY', 'XXZ', 'XZZ', 'YZZ', 'YYZ', 'XYZ']
    [42, 84, 126, 52, 44, 46, 62, 94, 86, 54],
    # 4: G
    # ['XXXX', 'YYYY', 'ZZZZ', 'XXXY', 'XXXZ', 'YYYX', 'YYYZ', 'ZZZX', 'ZZZY', 'XXYY',
    #  'XXZZ', 'YYZZ', 'XXYZ', 'YYXZ', 'ZZXY']
    [170, 340, 510, 172, 174, 338, 342, 506, 508, 180, 190, 350, 182, 334, 492],
]

def ao_info_from_gbasis(gbasis, cart_d=True, cart_f=True, cart_g=True):
    aoqnums = []
    ml_table = SHELL_ML_SPHERICAL.copy()
    if cart_d:
        ml_table[2] = SHELL_ML_CARTESIAN[2]
    if cart_f:
        ml_table[3] = SHELL_ML_CARTESIAN[3]
    if cart_g:
        ml_table[4] = SHELL_ML_CARTESIAN[4]
    for atom, atom_basis in enumerate(gbasis):
        counter = [0] * 5
        for shell, contractions in atom_basis:
            l = utils.SHELL_L[shell]
            counter[l] += 1
            n = counter[l]
            mls = ml_table[l]
            for ml in mls:
                q = (atom, n, l, ml)
                aoqnums.append(q)
    return aoqnums

class Molden(logfileparser.Logfile):
    """A Molcas log file."""

    def __init__(self, *args, **kwargs):
        super().__init__(logname="Molden", *args, **kwargs)

    def __str__(self):
        """Return a string repeesentation of the object."""
        return f"Molden file {self.filename}"

    def __repr__(self):
        """Return a representation of the object."""
        return f'Molden("{self.filename}")'

    def normalisesym(self, label):
        """Normalise the symmetries used by Molden.

        TODO: not implenemented
        """
        return label[0].upper() + label[1:]

    def after_parsing(self):
        if not hasattr(self, 'natom'):
            self.set_attribute('natom', len(self.atomcoords[0]))
        else:
            assert self.natom == len(self.atomcoords[0])
        if hasattr(self, 'gbasis'):
            assert len(self.gbasis) == self.natom
            aoqnums = ao_info_from_gbasis(self.gbasis, self.cart_d, self.cart_f, self.cart_g)
            self.set_attribute('aoqnums', aoqnums)
            self.set_attribute('nbasis', len(aoqnums))

    def before_parsing(self):
        self.cart_d = True
        self.cart_f = True
        self.cart_g = True

    def _parse_header(self, line):
        line = line.strip()
        section_header = line[1:line.rindex(']')]
        section_meta = None if line.endswith(']') else line.split()[-1]
        return section_header, section_meta

    def _section_lines(self, inputfile, line):
        if line.startswith('['):
            self._parse_header(line)
        for line in inputfile:
            if not line.startswith('['):
                yield line
            else:
                return
        self._eof = True

    def _iter_sections(self, inputfile, line):
        assert inputfile.last_line.startswith('[')
        self._eof = False
        while not self._eof:
            section_header, section_meta = self._parse_header(inputfile.last_line)
            section_lines = self._section_lines(inputfile, line)
            #  print('section:', section_header)
            yield section_header, section_meta, section_lines

    def extract(self, inputfile, line):
        """Extract information from the file object inputfile."""

        for section_header, section_meta, section_lines in self._iter_sections(inputfile, line):
            if section_header == 'Atoms':
                element_names = []
                numbers = []
                atomic_numbers = []
                coords = []
                for line in section_lines:
                    element_name, number, atomic_number, x, y, z = line.strip().split()
                    element_names.append(element_name)
                    numbers.append(int(number))
                    atomic_numbers.append(int(atomic_number))
                    coords.append((float(x), float(y), float(z)))
                if 'AU' in section_meta:
                    coords = numpy.asarray(coords)
                    coords = utils.convertor(coords, 'bohr', 'Angstrom')
                self.set_attribute('atomcoords', [coords])
                self.set_attribute('atomnos', atomic_numbers)

            elif section_header == 'N_Atoms':
                natom, = section_lines
                natom = int(natom)
                self.set_attribute('natom', natom)

            elif section_header == 'GTO':
                gbasis = []
                atom_data = []
                iatom = 1
                for line in section_lines:
                    if line.strip() == '':
                        if atom_data:
                            gbasis.append(atom_data)
                            atom_data = []
                    else:
                        parts = line.strip().split()
                        if len(parts) == 1:
                            continue
                        elif len(parts) == 2 and parts[1] == '0':
                            assert int(parts[0]) == iatom, parts
                            iatom += 1
                            continue
                        shell_label = parts[0].upper()
                        assert shell_label in "SPDFGHI", shell_label
                        num_primitives = int(parts[1])
                        primitives = []
                        for i in range(num_primitives):
                            primitive_line = next(inputfile)
                            primitive_line = primitive_line.replace('D', 'E')
                            primitive_parts = primitive_line.strip().split()
                            exponent = float(primitive_parts[0])
                            coefficient = float(primitive_parts[1])
                            primitives.append((exponent, coefficient))
                        atom_data.append((shell_label, primitives))
                if atom_data:
                    gbasis.append(atom_data)
                self.set_attribute('gbasis', gbasis)
                print(gbasis)

            elif section_header == 'MO':
                resulta = OrderedDict()
                resultb = OrderedDict()
                current_block = {}
                coeffs = []
                for line in section_lines:
                    if '=' in line:
                        if coeffs:
                            current_block['coeffs'] = coeffs
                            result = resultb if 'Spin' in current_block and current_block['Spin'] == 'Beta' else resulta
                            result[current_block['Sym']] = current_block
                            current_block = {}
                            coeffs = []
                        key, value = line.strip().split('=')
                        key = key.strip()
                        value = value.replace('D', 'E')
                        if key in ('Ene', 'Occup'):
                            value = float(value)
                        current_block[key] = value
                    else:
                        index, value = line.strip().split()
                        index = int(index)
                        value = value.replace('D', 'E')
                        assert index == len(coeffs) + 1
                        coeffs.append(float(value))
                if coeffs:
                    current_block['coeffs'] = coeffs
                    result = resultb if 'Spin' in current_block and current_block['Spin'] == 'Beta' else resulta
                    result[current_block['Sym']] = current_block
                if resultb:
                    keys = resulta.keys()
                    assert set(keys) == set(resultb.keys()), f'Keys not equal\n{keys}\n{resultb.keys()}'
                    resulta = [resulta[k] for k in keys]
                    resultb = [resultb[k] for k in keys]
                    mocoeffs = [[r['coeffs'] for r in resulta], [r['coeffs'] for r in resultb]]
                    moenergies = [[r['Ene'] for r in resulta], [r['Ene'] for r in resultb]]
                    #  nooccnos = [[r['Occup'] for r in resulta], [r['Occup'] for r in resultb]]
                    nooccnos = None
                else:
                    resulta = resulta.values()
                    mocoeffs = [[r['coeffs'] for r in resulta]]
                    moenergies = [[r['Ene'] for r in resulta]]
                    nooccnos = [r['Occup'] for r in resulta]
                self.set_attribute('mocoeffs', mocoeffs)
                moenergies = numpy.asarray(moenergies)
                moenergies = utils.convertor(moenergies, 'hartree', 'eV')
                self.set_attribute('moenergies', moenergies)
                if nooccnos is not None:
                    self.set_attribute('nooccnos', nooccnos)
                    self.set_attribute('nocoeffs', mocoeffs[0])
            if section_header[0].isdigit():
                if section_header == '5D':
                    self.cart_d = False
                    self.cart_f = False
                elif section_header == '7F':
                    self.cart_f = False
                elif section_header == '5D10F':
                    self.cart_d = False
                    self.cart_f = True
                elif section_header == '9G':
                    self.cart_g = False
                # consume the section lines
                list(section_lines)
            else:
                # consume the section lines
                list(section_lines)
