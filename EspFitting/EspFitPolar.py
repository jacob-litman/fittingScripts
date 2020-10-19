#!/usr/bin/env python
# General structure: subtract out unfit ESP (multipoles only) from each fit. Then, use MM probe & match ESP changes

import argparse
import numpy as np
import os
import re
from StructureXYZ import StructXYZ
from JMLUtils import eprint, version_file
from typing import List
import scipy.optimize

polar_patt = re.compile(r"^(polarize +)(\d+)( +)\d+\.\d+( +.+)$")


def edit_keyf_polarize(x: np.ndarray, from_fi: str, to_file: str = None, clobber: bool = False,
                       probe_types: List[int] = None) -> str:
    if to_file is None:
        to_file = from_fi
    if not clobber or from_fi == to_file:
        to_file = version_file(to_file)
    if probe_types is None:
        probe_types = [999]

    n_x = x.shape[0]
    polar_ctr = 0
    with open(from_fi, 'r') as r:
        with open(to_file, 'w') as w:
            for line in r:
                if line.startswith('polarize') and polar_ctr < n_x:
                    if int(m.group(2)) in probe_types:
                        eprint(f"Found polarize record with probe type {m.group(2)} as the {polar_ctr}'th place; "
                               f"probes expected to be last!\nRecord: {line}")
                        w.write(line)
                        continue
                    m = polar_patt.match(line)
                    # Will be more precision than the final key file.
                    w.write(f"{m.group(1)}{m.group(2)}{m.group(3)}{x[polar_ctr]:.10f}{m.group(2)}\n")
                    # TODO: Consider permitting tweaking of the Thole damping factor as well.
                    polar_ctr += 1
                    pass
                else:
                    w.write(line)
    return to_file


def cost_fun(x: np.ndarray, qm_diffs: List[np.ndarray], mm_refs: List[np.ndarray], probe_files: List[StructXYZ], key_files: List[str] = None,
             potential: str = 'potential', mol_pols: List[(float, float, float)] = None, wt_molpols: float = 0.0) -> float:
    n_probe = len(qm_diffs)
    assert n_probe == len(mm_refs) and n_probe == len(probe_files)
    if key_files is None:
        key_files = [pf.key_file for pf in probe_files]

    tot_sq_diff = 0
    for i in range(n_probe):
        keyf_i = edit_keyf_polarize(x, key_files[i])
        mm_diff = probe_files[i].get_esp(potential=potential, keyf=keyf_i) - mm_refs[i]
        # Assert that coordinates are equivalent. Can substitute in numpy.all_close
        assert np.array_equal(qm_diffs[i][:,0:2], mm_diff[:,0:2])
        qm_mm_diff = np.square(qm_diffs[i][:,3] - mm_diff[:,3])
        # Use mean rather than sum so as to normalize across different grid sizes.
        # TODO: Use same grid for everything, and/or focal-point weighting.
        tot_sq_diff += np.mean(qm_mm_diff)
    return tot_sq_diff


def main_inner(tinker_path: str = '', gauss_path: str = '', probe_types: List[int] = None):
    assert tinker_path is not None and gauss_path is not None
    if probe_types is None:
        probe_types = [999]
    # I will be fitting to <atom><ID>/qm_polarization.pot
    # Each round, I need to take QM_PR.xyz w/ modified key, get its potential, and subtract the probe potential.
    # Question: fit to every single grid point, or just RMS for that probe placement?

    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    if not gauss_path.endswith(os.sep) and gauss_path != '':
        gauss_path += os.sep
    formchk = gauss_path + "formchk"
    cubegen = gauss_path + "cubegen"

    probe_diffs = []
    structures = []
    mm_refs = []
    pot_cols = (1, 2, 3, 4)
    for pd in probe_dirs:
        # TODO: Consider changing dtype to np.float32 for improved performance. Assuming this is slow, anyways.
        # TODO: Consider using only column 4 to save memory (eliminates ability to check that X,Y,Z is identical).
        probe_diffs.append(np.genfromtxt(f"{pd}{os.sep}qm_polarization.pot", usecols=pot_cols, dtype=np.float64))
        mm_refs.append(np.genfromtxt(f"{pd}{os.sep}MM_REF.pot", usecols=pot_cols, dtype=np.float64))
        structures.append(StructXYZ(f"{pd}{os.sep}QM_PR.xyz", probe_types=probe_types,
                                    key_file=f"{pd}{os.sep}QM_PR.key"))

    opt_kwargs = {'qm_diffs': probe_diffs, 'mm_refs': mm_refs, 'probe_files': structures, 'potential': potential}


def main():
    parser = argparse.ArgumentParser()
    '''parser.add_argument('-a', dest='atom_index', type=int, required=True, help='Atom to place the probe near')
    parser.add_argument('-n', dest='probe_name', type=str, default='PC', help='Atom name to give the probe')
    parser.add_argument('-t', dest='probe_atype', type=int, default=999, help='Atom type to assign to the probe')
    parser.add_argument('-d', dest='distance', type=float, default=4.0, help='Distance to place the probe at')
    parser.add_argument('-w', dest='hydrogen_weight', type=float, default=0.4, help='Relative weighting for '
                                                                                    'hydrogen distances')
    parser.add_argument('-e', dest='exp', type=int, default=2, help='Exponent for the square of distance in target '
                                                                    'function')
    parser.add_argument('-k', dest='keyfile', type=str, default=None, help='Keyfile to use when saving to PDB')
    parser.add_argument('-x', dest='xyzpdb', type=str, default='xyzpdb', help='Name or full path of Tinker '
                                                                              'xyzpdb')
    parser.add_argument('-o', dest='outfile', type=str, default=None, help='XYZ file to output (if none: '
                                                                           'probe_atom<id>.xyz)')
    parser.add_argument('infile', nargs=1, type=str)'''
    

    args = parser.parse_args()

    main_inner()

if __name__ == "__main__":
    main()
