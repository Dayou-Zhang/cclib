# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""A writer for OpenMolcas Orb files."""

import numpy

from cclib.io import filewriter
from cclib.parser import utils

def generate_slices(nbas):
    cumsum = numpy.cumsum([0, *nbas])
    repeats = numpy.repeat(numpy.arange(len(nbas)), nbas)
    return zip(repeats+1, cumsum[repeats], cumsum[repeats+1])

def generate_mo_id(nbas):
    for n in nbas:
        yield from range(1, n+1)

class MolcasOrb(filewriter.Writer):
    """A writer for OpenMolcas Orb files."""

    def __init__(self, ccdata, title=None, *args, **kwargs):
        """Initialize the MolcasOrb writer object.

        Inputs:
          ccdata - An instance of ccData, parse from a logfile.
        """
        super().__init__(ccdata, *args, **kwargs)

        self.required_attrs = ('nbasis', 'mocoeffs')
        self.title = title

    def generate_repr(self):
        data = self.ccdata
        nbasis = data.nbasis
        title = self.title
        try:
            nbas = data.metadata['nbas']
            nsym = len(nbas)
        except (AttributeError, KeyError):
            nbas = [nbasis]
            nsym = 1
        nbas_str = ''.join('%8d'%i for i in nbas)
        ao_splitter = numpy.cumsum(nbas)[:-1]
        ao_slices = generate_slices(nbas)
        mo_id = generate_mo_id(nbas)
        if title is None:
            title = 'Untitled Orbital'
        lines = []
        with numpy.printoptions(threshold=numpy.inf):
            lines.append(f'''#INPORB 2.2
#INFO
* {title}
       0       {nsym}       0
{nbas_str}
{nbas_str}
#ORB
''')
            for i in data.mocoeffs[0]:
                s = next(ao_slices)
                n = next(mo_id)
                lines.append(f"* ORBITAL{s[0]:5d}{n:5d}\n ")
                lines.append(numpy.array2string(i[s[1]:s[2]], max_line_width=115, formatter={'float_kind':lambda x: "%21.14E" % x})[1:-1])
                lines.append('\n')

            try:
                mo_occ = data.nooccnos
            except AttributeError:
                mo_occ = numpy.zeros(nbasis)
            lines.append('#OCC\n* OCCUPATION NUMBERS\n')
            for i in numpy.split(mo_occ, ao_splitter):
                if i.size == 0:
                    continue
                lines.append(' ')
                lines.append(numpy.array2string(i, max_line_width=115, formatter={'float_kind':lambda x: "%21.14E" % x})[1:-1])
                lines.append('\n')

            try:
                mo_energy = data.moenergies[0]
            except AttributeError:
                mo_energy = numpy.zeros(nbasis)
            mo_energy = utils.convertor(mo_energy, 'eV', 'hartree')
            lines.append('#ONE\n* ONE ELECTRON ENERGIES\n')
            for i in numpy.split(mo_energy, ao_splitter):
                if i.size == 0:
                    continue
                lines.append(' ')
                lines.append(numpy.array2string(i, max_line_width=122, formatter={'float_kind':lambda x: "%11.4E" % x})[1:-1])
                lines.append('\n')
        return ''.join(lines)

