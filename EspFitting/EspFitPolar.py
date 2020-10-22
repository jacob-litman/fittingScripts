#!/usr/bin/env python
# General structure: subtract out unfit ESP (multipoles only) from each fit. Then, use MM probe & match ESP changes

import argparse
import numpy as np
import os
import re
import sys
from StructureXYZ import StructXYZ
from JMLUtils import eprint, version_file
from typing import Sequence
import scipy.optimize

polar_patt = re.compile(r"^(polarize +)(\d+)( +)(\d+\.\d+)( +.+)$")


def read_initial_polarize(from_fi: str, probe_types: Sequence[int] = None) -> np.ndarray:
    if probe_types is None:
        probe_types = [999]
    #polar_types = []  # Would contain the atom types; not necessary yet.
    polarizabilities = []
    with open(from_fi, 'r') as r:
        for line in r:
            if line.startswith('polarize'):
                m = polar_patt.match(line)
                assert m
                atype = int(m.group(2))
                if atype not in probe_types:
                    #polar_types.append(atype)
                    polarizabilities.append(float(m.group(4)))
    return np.array(polarizabilities)


def edit_keyf_polarize(x: np.ndarray, from_fi: str, to_file: str = None, clobber: bool = False,
                       probe_types: Sequence[int] = None) -> str:
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
                    m = polar_patt.match(line)
                    assert m
                    if int(m.group(2)) in probe_types:
                        eprint(f"Found polarize record with probe type {m.group(2)} as the {polar_ctr}'th place; "
                               f"probes expected to be last!\nRecord: {line}")
                        w.write(line)
                        continue
                    # Will be more precision than the final key file.
                    w.write(f"{m.group(1)}{m.group(2)}{m.group(3)}{x[polar_ctr]:.10f}  {m.group(5)}\n")
                    # TODO: Consider permitting tweaking of the Thole damping factor as well.
                    polar_ctr += 1
                    pass
                else:
                    w.write(line)
    return to_file


def cost_fun(x: np.ndarray, qm_diffs: Sequence[np.ndarray], mm_refs: Sequence[np.ndarray], probe_files: Sequence[StructXYZ], key_files: Sequence[str] = None,
             potential: str = 'potential', mol_pols: np.ndarray = None, wt_molpols: float = 0.0) -> float:
    n_probe = len(qm_diffs)
    assert n_probe == len(mm_refs) and n_probe == len(probe_files)
    if key_files is None:
        key_files = [pf.key_file for pf in probe_files]
    curr_keyfiles = key_files.copy()

    tot_sq_diff = 0
    for i in range(n_probe):
        keyf_i = edit_keyf_polarize(x, key_files[i], to_file=curr_keyfiles[i])
        curr_keyfiles[i] = keyf_i
        mm_diff = probe_files[i].get_esp(potential=potential, keyf=keyf_i)
        mm_diff[:,3] -= mm_refs[i][:,3]
        # eprint(f"mm diff: {mm_diff}")
        # eprint(f"qm diff: {qm_diffs[i]}")
        # Assert that coordinates are equivalent. Can substitute in numpy.all_close
        assert np.array_equal(qm_diffs[i][:,0:2], mm_diff[:,0:2])
        qm_mm_diff = np.square(qm_diffs[i][:,3] - mm_diff[:,3])
        # Use mean rather than sum so as to normalize across different grid sizes.
        # TODO: Use same grid for everything, and/or focal-point weighting.
        tot_sq_diff += np.mean(qm_mm_diff)
    return tot_sq_diff


def main_inner(tinker_path: str = '', probe_types: Sequence[int] = None):
    assert tinker_path is not None
    if probe_types is None:
        probe_types = [999]
    # I will be fitting to <atom><ID>/qm_polarization.pot
    # Each round, I need to take QM_PR.xyz w/ modified key, get its potential, and subtract the probe potential.
    # Question: fit to every single grid point, or just RMS for that probe placement?

    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    probe_diffs = []
    structures = []
    mm_refs = []
    pot_cols = (1, 2, 3, 4)
    for pd in probe_dirs:
        # TODO: Consider changing dtype to np.float32 for improved performance. Assuming this is slow, anyways.
        # TODO: Consider using only column 4 to save memory (eliminates ability to check that X,Y,Z is identical).
        probe_diffs.append(np.genfromtxt(f"{pd}{os.sep}qm_polarization.pot", usecols=pot_cols, skip_header=1,
                                         dtype=np.float64))
        mm_refs.append(np.genfromtxt(f"{pd}{os.sep}MM_REF_BACK.pot", usecols=pot_cols, skip_header=1, dtype=np.float64))
        structures.append(StructXYZ(f"{pd}{os.sep}QM_PR.xyz", probe_types=probe_types,
                                    key_file=f"{pd}{os.sep}QM_PR.key"))

    x = read_initial_polarize(structures[0].key_file, probe_types=probe_types)
    opt_kwargs = {'qm_diffs': probe_diffs, 'mm_refs': mm_refs, 'probe_files': structures, 'potential': potential}
    verbosity = 2
    bounds = (0, np.inf)
    ls_result = scipy.optimize.least_squares(cost_fun, x, jac='3-point', kwargs=opt_kwargs, verbose=verbosity,
                                             bounds=bounds)
    eprint(f'\n\nLeast squares result:\n{ls_result}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='tinker_path', type=str, default='', help='Path to Tinker executables')
    parser.add_argument('-p', dest='probe_type', type=int, default=999, help='Probe atom type')
    args = parser.parse_args()
    
    main_inner(tinker_path=args.tinker_path, probe_types=[args.probe_type])

if __name__ == "__main__":
    main()
