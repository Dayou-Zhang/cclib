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
        if title is None:
            title = 'Untitled Orbital'
        lines = []
        with numpy.printoptions(threshold=numpy.inf):
            lines.append(f'''#INPORB 2.2
#INFO
* {title}
       0       1       0
{nbasis:8d}
{nbasis:8d}
#ORB
''')
            for n, i in enumerate(data.mocoeffs[0], 1):
                lines.append(f"* ORBITAL    1{n:5d}\n ")
                lines.append(numpy.array2string(i, max_line_width=115, formatter={'float_kind':lambda x: "%21.14E" % x})[1:-1])
                lines.append('\n')

            try:
                mo_occ = data.nooccnos
            except AttributeError:
                mo_occ = numpy.zeros(nbasis)
            lines.append('#OCC\n* OCCUPATION NUMBERS\n ')
            lines.append(numpy.array2string(mo_occ, max_line_width=115, formatter={'float_kind':lambda x: "%21.14E" % x})[1:-1])
            lines.append('\n')

            try:
                mo_energy = data.moenergies[0]
            except AttributeError:
                mo_energy = numpy.zeros(nbasis)
            mo_energy = utils.convertor(mo_energy, 'eV', 'hartree')
            lines.append('#ONE\n* ONE ELECTRON ENERGIES\n ')
            lines.append(numpy.array2string(mo_energy, max_line_width=122, formatter={'float_kind':lambda x: "%11.4E" % x})[1:-1])
            lines.append('\n')
        return ''.join(lines)

