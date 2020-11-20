#!/usr/bin/env python
# General structure: subtract out unfit ESP (multipoles only) from each fit. Then, use MM probe & match ESP changes

import argparse
import os
import re
import sys
from typing import Sequence, Mapping

import numpy as np
import scipy.optimize

from JMLUtils import eprint, get_probe_dirs, version_file
from StructureXYZ import StructXYZ

polar_patt = re.compile(r"^(polarize +)(\d+)( +)(\d+\.\d+)( +.+)$")
DEFAULT_POLARTYPE_FI = 'polartype-define.txt'


def edit_keyf_polarize(polar_type_map: Mapping[str, float], from_fi: str, to_file: str, pr_xyz: StructXYZ,
                       precis: int = 10):
    assert from_fi != to_file
    probe_types = pr_xyz.probe_types
    if probe_types is None:
        probe_types = [999]

    n_polar = pr_xyz.polarization_types.shape[0]
    ptypes = pr_xyz.polarization_types[:0]
    polar_ctr = 0

    with open(from_fi, 'r') as r:
        with open(to_file, 'w') as w:
            for line in r:
                if line.startswith('polarize') and polar_ctr < n_polar:
                    m = polar_patt.match(line)
                    assert m
                    if int(m.group(2)) in probe_types:
                        eprint(f"Found polarize record with probe type {m.group(2)} as the {polar_ctr}'th place; "
                               f"probes expected to be last!\nRecord: {line}")
                        w.write(line)
                        continue
                    polarizability = polar_type_map[ptypes[polar_ctr]]
                    w.write(f"{m.group(1)}{m.group(2)}{m.group(3)}{polarizability:.{precis}f}  {m.group(5)}\n")
                    # TODO: Consider permitting tweaking of the Thole damping factor as well.
                    polar_ctr += 1
                else:
                    w.write(line)


def cost_fun_residual(x: np.ndarray, qm_polars: np.ndarray, mm_refs: np.ndarray, probe_xyzs: np.ndarray,
                      probe_npoints: np.ndarray, out_keys: np.ndarray, in_keys: np.ndarray, polarize_defs: np.ndarray,
                      probe_weights: np.ndarray = None, potential: str = 'potential') -> np.ndarray:
    # TODO: Weight these things properly.
    n_points = qm_polars.shape[0]
    assert qm_polars.shape[1] == 4 and qm_polars.ndim == 2
    assert mm_refs.shape == qm_polars.shape
    polar_mapping = {}
    for i, pdef in enumerate(polarize_defs):
        polar_mapping[polarize_defs[i][0]] = x[i]

    n_probes = probe_xyzs.size
    if probe_weights is None:
        probe_weights = np.ones(n_probes, dtype=np.float64)
    assert n_probes == out_keys.size and n_probes == in_keys.size
    offset = 0
    residuals = np.empty(n_points, dtype=np.float64)
    for i in range(n_probes):
        n_this = probe_npoints[i]
        weight_point = probe_weights[i] / n_this
        off_this = offset + n_this + 1
        edit_keyf_polarize(polar_mapping, in_keys[i], out_keys[i], probe_xyzs[i])
        mm_diff = probe_xyzs[i].get_esp(potential=potential, keyf=out_keys[i])
        mm_diff[:,3] -= mm_refs[offset:off_this,3]
        assert np.array_equal(qm_polars[offset:off_this, 0:2], mm_diff[:, 0:2])
        qm_mm_diff = qm_polars[offset:off_this, 3] - mm_diff[:, 3]
        residuals[offset:off_this] = (qm_mm_diff * weight_point)

    sys.stderr.flush()
    sys.stdout.flush()
    return residuals


def cost_fun(x: np.ndarray, qm_polars: np.ndarray, mm_refs: np.ndarray, probe_xyzs: np.ndarray,
                      probe_npoints: np.ndarray, out_keys: np.ndarray, in_keys: np.ndarray, polarize_defs: np.ndarray,
                      probe_weights: np.ndarray = None, potential: str = 'potential') -> float:
    residuals = cost_fun_residual(x, qm_polars, mm_refs, probe_xyzs, probe_npoints, out_keys, in_keys, polarize_defs,
                                  probe_weights, potential)
    assert len(residuals) == 1
    return np.mean(np.square(residuals))[0]


def get_pot_points(pot_fi: str) -> int:
    n_pts = -1
    with open(pot_fi, 'r') as r:
        for _ in r:
            n_pts += 1
    return n_pts


def main_inner(molec_fis: Sequence[str], tinker_path: str = '', probe_types: Sequence[int] = None, tol: float = None,
               min_strat: str = 'least_squares', residual: bool = False, maxiter: int = 250, x: np.ndarray = None,
               polartype_fi: str = DEFAULT_POLARTYPE_FI):
    assert molec_fis is not None and len(molec_fis) > 0
    assert tinker_path is not None
    if probe_types is None:
        probe_types = [999]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    least_sq = (min_strat.lower() == 'least_squares') or (min_strat.lower() == 'ls')
    polar_defs = np.genfromtxt(polartype_fi)

    ref_mols = []
    probe_dirs = []
    probe_mols = []
    in_keys = []
    out_keys = []
    pot_cols = [1, 2, 3, 4]
    num_pr_points = []

    for fi in molec_fis:
        this_struct = StructXYZ(fi, load_polar_types=True)
        this_dir = os.path.dirname(fi)
        ref_mols.append(this_struct)
        these_pdirs = get_probe_dirs(this_dir)
        probe_dirs.append(these_pdirs)
        this_probes = [StructXYZ(os.path.join(pd, "QM_PR.xyz"), probe_types=probe_types, load_polar_types=True)
                       for pd in these_pdirs]
        probe_mols.append(this_probes)
        this_pr_points = []
        for pd in these_pdirs:
            this_pr_points.append(get_pot_points(os.path.join(pd, 'qm_polarization.pot')))
        num_pr_points.append(this_pr_points)

    n_mols = len(ref_mols)
    ref_mols = np.array(ref_mols)
    molec_points = np.zeros_like(ref_mols)
    tot_points = 0
    n_probes = 0
    for i in range(n_mols):
        len_i = len(num_pr_points[i])
        for j in range(len_i):
            molec_points[i] += num_pr_points[i][j]
        tot_points += molec_points[i]
        n_probes += len_i

    qm_polarizations = np.empty([tot_points, 4])
    mm_ref_potentials = np.empty([tot_points, 4])
    probe_weights = np.empty(n_probes, dtype=np.float64)
    offset = 0
    for i, fi in enumerate(molec_fis):
        these_pdirs = probe_dirs[i]
        this_probes = probe_mols[i]
        probe_weights[i] = 1.0 / len(these_pdirs)
        for j, pd in enumerate(these_pdirs):
            this_pol = np.genfromtxt(os.path.join(pd, 'qm_polarization.pot'), usecols=pot_cols, skip_header=1,
                                     dtype=np.float64)
            n_this = num_pr_points[i][j]
            off_this = offset + n_this + 1
            assert this_pol.shape[0] == n_this
            qm_polarizations[offset:off_this, :] = this_pol

            this_pot = np.genfromtxt(os.path.join(pd, 'MM_REF_BACK.pot'), usecols=pot_cols, skip_header=1,
                                     dtype=np.float64)
            assert this_pot.shape[0] == n_this
            mm_ref_potentials[offset:off_this, :] = this_pot
        this_keyfs = [pr.key_file for pr in this_probes]
        in_keys.append(this_keyfs)
        out_keys.append([version_file(kf) for kf in this_keyfs])

    # Final flattening of key arrays.
    probe_mols = np.reshape(np.array(probe_mols, dtype=StructXYZ), newshape=-1)
    probe_npoints = np.reshape(num_pr_points, dtype=np.int32, newshape=-1)

    opt_args = (qm_polarizations, mm_ref_potentials, probe_mols, probe_npoints, out_keys, in_keys, polar_defs,
                probe_weights, potential)

    if tol is None:
        if least_sq:
            tol = 1E-5
        else:
            tol = 1E-6

    if x is None:
        x = polar_defs[:,1]
        eprint("Read input polarizabilities: beginning optimization.")

    if least_sq:
        bounds = (0, np.inf)
        if residual:
            func = cost_fun_residual
        else:
            func = cost_fun
        ls_result = scipy.optimize.least_squares(func, x, jac='3-point', args=opt_args, verbose=2, bounds=bounds,
                                                 xtol=tol)
        eprint(f'\n\nLeast squares result:\n{ls_result}')
    else:
        raise ValueError("Non-least-squares optimization not yet implemented.")


def get_molec_files(opt_info_fi: str) -> Sequence[str]:
    files_out = []
    with open(opt_info_fi, 'r') as r:
        for line in r:
            line = re.sub(r'#.+', '', line).strip()
            if len(line) > 0:
                if os.path.exists(line) and os.path.isfile(line):
                    files_out.append(line)
    return files_out


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
    parser.add_argument('optimize_info', type=str, nargs=1, default='molecules.txt',
                        help='File containing paths to molecules (i.e. path/to/QM_REF.xyz)')

    args = parser.parse_args()
    mfis = get_molec_files(args.optimize_info)
    assert len(mfis) > 0

    main_inner(mfis, args.tinker_path, probe_types=[args.probe_type], tol=args.tol, min_strat=args.min,
               residual=args.residual, maxiter=args.maxiter, x=None)


if __name__ == "__main__":
    main()
