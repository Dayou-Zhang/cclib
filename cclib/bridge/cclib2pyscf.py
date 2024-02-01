# -*- coding: utf-8 -*-
#
# Copyright (c) 2019, the cclib development team
#
# This file is part of cclib (http://cclib.github.io) and is distributed under
# the terms of the BSD 3-Clause License.

"""Bridge for using cclib data in PySCF (https://github.com/pyscf/pyscf)."""

from cclib.parser.utils import find_package, PeriodicTable, convertor
import numpy as np


class MissingAttributeError(Exception):
    pass

_found_pyscf = find_package("pyscf")
if _found_pyscf:
    from pyscf import gto
    from pyscf.lib.parameters import REAL_SPHERIC, ANGULARMAP


def _check_pyscf(found_pyscf):
    if not found_pyscf:
        raise ImportError("You must install `pyscf` to use this function")


def makepyscf(data, charge=None, mult=None):
    """Create a Pyscf Molecule."""
    _check_pyscf(_found_pyscf)
    inputattrs = data.__dict__
    required_attrs = {"atomcoords", "atomnos"}
    missing = [x for x in required_attrs if not hasattr(data, x)]
    if missing:
        missing = " ".join(missing)
        raise MissingAttributeError(
            f"Could not create pyscf molecule due to missing attribute: {missing}"
        )

    if charge is None:
        if "charge" in inputattrs:
            charge = data.charge
        else:
            charge = 0

    if mult is None:
        if "mult" in inputattrs:
            mult = data.mult
        else:
            mult = 1

    mol = gto.Mole(
        atom=[
            [f"{data.atomnos[i]}", data.atomcoords[-1][i]] for i in range(data.natom)
        ],
        unit="Angstrom",
        charge=charge,
        spin=mult-1
    )
    inputattr = data.__dict__
    pt = PeriodicTable()
    if "gbasis" in inputattr:
        basis = {}  # object for internal PySCF format
        uatoms, uatoms_idx = np.unique(
            data.atomnos, return_index=True
        )  # find unique atoms
        for idx, i in enumerate(uatoms_idx):
            curr_atom_basis = data.gbasis[i]
            for jdx, j in enumerate(curr_atom_basis):
                curr_l = j[0]
                curr_e_prim = j[1]
                new_list = [ANGULARMAP[curr_l.lower()]]
                new_list += curr_e_prim
                if not f"{pt.element[uatoms[idx]]}" in basis:
                    basis[f"{pt.element[uatoms[idx]]}"] = [new_list]
                else:
                    basis[f"{pt.element[uatoms[idx]]}"].append(new_list)
        mol.basis = basis
    mol.build()
    return mol

def makepyscf_mos(ccdata,mol):
    """
    Returns pyscf formatted MO properties from a cclib object.
    Parameters
    ---
    ccdata: cclib object
        cclib object from parsed output
    mol: pyscf Molecule object
       molecule object that must contain the mol.basis attribute

    Returns
    ----
    mo_coeff : n_spin x nmo x nao ndarray
        molecular coeffcients, unnormalized according to pyscf standards
    mo_occ : array
        molecular orbital occupation
    mo_syms : array
       molecular orbital symmetry labels
    mo_energies: array
        molecular orbital energies in units of Hartree
    """
    inputattrs = ccdata.__dict__
    if "mocoeffs" in inputattrs:
        mol.build()
        s = mol.intor('int1e_ovlp')
        if np.shape(ccdata.mocoeffs)[0] == 1:
            mo_coeffs = np.einsum('i,ij->ij', np.sqrt(1/s.diagonal()), ccdata.mocoeffs[0].T)
            mo_occ = np.zeros(ccdata.nmo)
            mo_occ[:ccdata.homos[0]+1] = 2
            mo_energies = convertor(np.array(ccdata.moenergies),"eV","hartree")
            if hasattr(ccdata, 'mosyms'):
                mo_syms = ccdata.mosyms
            else:
                mo_syms = np.full_like(ccdata.moenergies, 'A', dtype=str)

        elif np.shape(ccdata.mocoeffs)[0] == 2:
            mo_coeff_a = np.einsum('i,ij->ij', np.sqrt(1/s.diagonal()), ccdata.mocoeffs[0].T)
            mo_coeff_b = np.einsum('i,ij->ij', np.sqrt(1/s.diagonal()), ccdata.mocoeffs[1].T)
            mo_occ = np.zeros((2,ccdata.nmo))
            mo_occ[0,:ccdata.homos[0]+1] = 1
            mo_occ[1,:ccdata.homos[1]+1] = 1
            mo_coeffs = np.array([mo_coeff_a,mo_coeff_b])
            mo_energies = convertor(np.array(ccdata.moenergies),"eV","hartree")
            if hasattr(ccdata, 'mosyms'):
                mo_syms = ccdata.mosyms
            else:
                mo_syms = np.full_like(ccdata.moenergies, 'A', dtype=str)
    return mo_coeffs, mo_occ, mo_syms, mo_energies

# pyscf: xyz -> -1 0 1 , cclib: xyz -> 1 -1 0
P_ML_MAP = (-1, 0, 1)
def get_aoqnums(mol):
    if mol.cart:
        raise ValueError('Cartesian basis not implemented')
    atom_ids = []
    n_list = []
    l_list = []
    ml_list = []
    ao_labels = mol.ao_labels(False)
    for iatom, _, ilabel, iang in ao_labels:
        atom_ids.append(iatom)
        l = ANGULARMAP[ilabel[-1]]
        n = int(ilabel[:-1]) - l
        ml = REAL_SPHERIC[l].index(iang) - l
        if l == 1:
            ml = P_ML_MAP[ml]
        n_list.append(n)
        l_list.append(l)
        ml_list.append(ml)
    return np.array([atom_ids, n_list, l_list, ml_list]).T

del find_package
