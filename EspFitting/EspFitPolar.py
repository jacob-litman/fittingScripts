#!/usr/bin/env python
# General structure: subtract out unfit ESP (multipoles only) from each fit. Then, use MM probe & match ESP changes

import argparse
import os
import re
import sys
from typing import Sequence

import numpy as np
import scipy.optimize

from JMLUtils import eprint, version_file
from StructureXYZ import StructXYZ

polar_patt = re.compile(r"^(polarize +)(\d+)( +)(\d+\.\d+)( +.+)$")


def read_initial_polarize(from_fi: str, probe_types: Sequence[int] = None) -> np.ndarray:
    if probe_types is None:
        probe_types = [999]
    # polar_types = []  # Would contain the atom types; not necessary yet.
    polarizabilities = []
    with open(from_fi, 'r') as r:
        for line in r:
            if line.startswith('polarize'):
                m = polar_patt.match(line)
                assert m
                atype = int(m.group(2))
                if atype not in probe_types:
                    # polar_types.append(atype)
                    polarizabilities.append(float(m.group(4)))
    return np.array(polarizabilities)


def edit_keyf_polarize(x: np.ndarray, from_fi: str, to_file: str, probe_types: Sequence[int] = None):
    '''if to_file is None:
        to_file = from_fi
    if not clobber or from_fi == to_file:
        to_file = version_file(to_file)'''
    assert from_fi != to_file
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


def cost_fun_residual(x: np.ndarray, qm_diffs: Sequence[np.ndarray], mm_refs: Sequence[np.ndarray],
                      probe_files: Sequence[StructXYZ], to_files: Sequence[str], key_files: Sequence[str] = None,
                      potential: str = 'potential', mol_pols: np.ndarray = None, wt_molpols: float = 0.0) -> np.ndarray:
    n_probe = len(qm_diffs)
    assert n_probe == len(mm_refs) and n_probe == len(probe_files)
    if key_files is None:
        key_files = [pf.key_file for pf in probe_files]

    residuals = []
    for i in range(n_probe):
        edit_keyf_polarize(x, key_files[i], to_files[i])
        mm_diff = probe_files[i].get_esp(potential=potential, keyf=to_files[i])
        mm_diff[:, 3] -= mm_refs[i][:, 3]
        # Assert that coordinates are equivalent. Can substitute in numpy.all_close
        assert np.array_equal(qm_diffs[i][:, 0:2], mm_diff[:, 0:2])
        qm_mm_diff = qm_diffs[i][:, 3] - mm_diff[:, 3]
        residuals.append(qm_mm_diff)
    sys.stderr.flush()
    sys.stdout.flush()
    return np.concatenate(residuals)


def cost_fun(x: np.ndarray, qm_diffs: Sequence[np.ndarray], mm_refs: Sequence[np.ndarray],
             probe_files: Sequence[StructXYZ], to_files: Sequence[str], key_files: Sequence[str] = None,
             potential: str = 'potential', mol_pols: np.ndarray = None, wt_molpols: float = 0.0) -> float:
    n_probe = len(qm_diffs)
    assert n_probe == len(mm_refs) and n_probe == len(probe_files)
    if key_files is None:
        key_files = [pf.key_file for pf in probe_files]

    tot_sq_diff = 0
    for i in range(n_probe):
        edit_keyf_polarize(x, key_files[i], to_files[i])
        mm_diff = probe_files[i].get_esp(potential=potential, keyf=to_files[i])
        mm_diff[:, 3] -= mm_refs[i][:, 3]
        # Assert that coordinates are equivalent. Can substitute in numpy.all_close
        assert np.array_equal(qm_diffs[i][:, 0:2], mm_diff[:, 0:2])
        qm_mm_diff = qm_diffs[i][:, 3] - mm_diff[:, 3]
        # Use mean rather than sum so as to normalize across different grid sizes.
        # TODO: Use same grid for everything, and/or focal-point weighting.
        tot_sq_diff += np.mean(np.square(qm_mm_diff))
    sys.stderr.flush()
    sys.stdout.flush()
    return tot_sq_diff


def main_inner(tinker_path: str = '', probe_types: Sequence[int] = None, tol: float = None, x: Sequence[float] = None,
               min_strat: str = 'least_squares', residual: bool = False, maxiter: int = 250):
    assert tinker_path is not None
    if probe_types is None:
        probe_types = [999]

    least_sq = (min_strat.lower() == 'least_squares') or (min_strat.lower() == 'ls')

    if tol is None:
        if least_sq:
            tol = 1E-5
        else:
            tol = 1E-6

    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    probe_diffs = []
    structures = []
    mm_refs = []
    pot_cols = (1, 2, 3, 4)
    for pd in probe_dirs:
        eprint(f"Reading information for probe {pd}")
        # TODO: Consider changing dtype to np.float32 for improved performance. Assuming this is slow, anyways.
        # TODO: Consider using only column 4 to save memory (eliminates ability to check that X,Y,Z is identical).
        probe_diffs.append(np.genfromtxt(f"{pd}{os.sep}qm_polarization.pot", usecols=pot_cols, skip_header=1,
                                         dtype=np.float64))
        mm_refs.append(np.genfromtxt(f"{pd}{os.sep}MM_REF_BACK.pot", usecols=pot_cols, skip_header=1, dtype=np.float64))
        structures.append(StructXYZ(f"{pd}{os.sep}QM_PR.xyz", probe_types=probe_types,
                                    key_file=f"{pd}{os.sep}QM_PR.key"))
    key_files = [pf.key_file for pf in structures]
    out_keys = [version_file(kf) for kf in key_files]

    if x is None:
        x = read_initial_polarize(structures[0].key_file, probe_types=probe_types)

    cost_fun_args = (probe_diffs, mm_refs, structures, out_keys, key_files, potential, None, 0.0)
    eprint("Read input polarizabilities: beginning optimization.")
    if least_sq:
        bounds = (0, np.inf)
        if residual:
            func = cost_fun_residual
        else:
            func = cost_fun
        ls_result = scipy.optimize.least_squares(func, x, jac='3-point', args=cost_fun_args, verbose=2,
                                                 bounds=bounds, xtol=tol)
        eprint(f'\n\nLeast squares result:\n{ls_result}')
    else:
        bounds = [(0, None) for xi in x]
        opt_args = {'disp': 1, 'maxiter': maxiter, 'gtol': tol, 'maxcor': 25}
        # TODO: Not hardcode in L-BFGS-B.
        min_result = scipy.optimize.minimize(cost_fun, x, args=cost_fun_args, options=opt_args, method='L-BFGS-B',
                                             jac='3-point', bounds=bounds)
        eprint(f'\n\nMinimization result:\n{min_result}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tinkerpath', dest='tinker_path', type=str, default='', help='Path to Tinker '
                                                                                             'executables')
    parser.add_argument('-p', '--probetype', dest='probe_type', type=int, default=999, help='Probe atom type')
    parser.add_argument('-e', '--tol', dest='tol', type=float, default=None,
                        help='Cease optimization at this tolerance (otherwise algorithm-dependent default)')
    # parser.add_argument('-x', type=Sequence[float], default=None, help='Manually specify starting polarizabilities')
    parser.add_argument('-m', '--minstrategy', dest='minstrategy', default='least_squares', type=str,
                        help='Choose optimization method from least_squares or BFGS')
    parser.add_argument('-r', '--residual', dest='residual', action='store_true',
                        help='Use residual rather than the target function for least-squares optimization '
                             '(least_squares only!)')
    parser.add_argument('-i', '--maxiter', dest='max_iter', type=int, default=250, help='Maximum iterations of the '
                                                                                        'optimizer')

    args = parser.parse_args()

    main_inner(tinker_path=args.tinker_path, probe_types=[args.probe_type], tol=args.tol, min_strat=args.minstrategy,
               residual=args.residual, maxiter=args.max_iter)


if __name__ == "__main__":
    main()
