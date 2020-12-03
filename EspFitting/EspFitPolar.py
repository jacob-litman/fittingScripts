import argparse
import re
import os
from typing import Sequence, Mapping
from JMLUtils import eprint, get_probe_dirs, version_file
from os.path import join
import StructureXYZ
import numpy as np
import scipy.optimize
import time
import concurrent.futures
import sys

DEFAULT_TOL = 1E-4
DEFAULT_MAX_ITER = 1000
required_subdir_files = ['qm_polarization.pot', 'MM_REF_BACK.pot', 'QM_PR.xyz', 'QM_PR.key']
pot_cols = [1, 2, 3, 4]
polarize_patt = re.compile(r'^(polarize\s+)(\d+\s+)(\d+\.\d+)(\s+.+)$')


def edit_keyfile(in_key: str, out_key: str, x: np.ndarray, atype_map: Mapping[int, int], precis: int = 12):
    with open(in_key, 'r') as r:
        with open(out_key, 'w') as w:
            for line in r:
                if line.startswith("polarize"):
                    m = polarize_patt.match(line)
                    assert m
                    atype = int(m.group(2).strip())
                    if atype in atype_map:
                        out_polar = x[atype_map[atype]]
                        w.write(f"{m.group(1)}{m.group(2)}{out_polar:.{precis}f}{m.group(4).rstrip()}\n")
                    else:
                        w.write(line)
                else:
                    w.write(line)


def cost_function_sequential(x: np.ndarray, probe_mols: Sequence[StructureXYZ.StructXYZ], out_keys: Sequence[str],
                  atype_maps: Sequence[Mapping[int, int]], potential: str, targets: Sequence[np.ndarray]) -> float:
    sys.stdout.flush()
    sys.stderr.flush()
    del_time = -time.time()
    tot_cost = 0
    for i, pm in enumerate(probe_mols):
        tot_cost += cost_interior(x, pm, out_keys[i], atype_maps[i], potential, targets[i])

    del_time += time.time()
    eprint(f"Function evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
    sys.stdout.flush()
    sys.stderr.flush()
    return tot_cost


def cost_function(x: np.ndarray, probe_mols: Sequence[StructureXYZ.StructXYZ], out_keys: Sequence[str],
                  atype_maps: Sequence[Mapping[int, int]], potential: str, targets: Sequence[np.ndarray],
                  executor: concurrent.futures.Executor) -> float:
    sys.stdout.flush()
    sys.stderr.flush()
    del_time = -time.time()
    tot_cost = 0
    costbundles = []
    for i, pm in enumerate(probe_mols):
        costbundles.append((x, pm, out_keys[i], atype_maps[i], potential, targets[i]))

    futures = {executor.submit(cost_interior, *costbundle) for costbundle in costbundles}
    for future in concurrent.futures.as_completed(futures):
        tot_cost += future.result()

    del_time += time.time()
    eprint(f"Function evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
    sys.stdout.flush()
    sys.stderr.flush()
    return tot_cost


def cost_interior(x: np.ndarray, pm: StructureXYZ.StructXYZ, outkey: str, atype_map: Mapping[int, int], potential: str,
                  target: np.ndarray) -> float:
    edit_keyfile(pm.key_file, outkey, x, atype_map)
    mm_potential = pm.get_esp(potential, outkey)
    assert np.array_equal(mm_potential[:, 0:2], target[:, 0:2])
    mm_potential[:, 3] -= target[:, 3]
    return np.average(np.square(mm_potential[:, 3]))


def main_inner(mfis: Sequence[str], ptype_names: np.ndarray, initial_x: np.ndarray, tinker_path: str = '',
               probe_types: Sequence[int] = None, tol: float = DEFAULT_TOL, maxiter: int = DEFAULT_MAX_ITER):
    # 1D lists associated with reference molecules
    ref_mols = []
    # 1D pre-flattened lists associated with probe molecules. Some are effectively duplicates based on reference molecule.
    targets = []
    probe_mols = []
    grid_points = []
    out_keys = []
    atypes_to_x_index = [] # 1D list of Dict[int,int] mapping that molecule's atom types to an index in x.

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    for fi in mfis:
        rdir = os.path.dirname(fi)
        rmol = StructureXYZ.StructXYZ(fi, probe_types=probe_types, load_polar_types=True)
        ref_mols.append(rmol)
        pdirs_i = []
        atype_map = dict()
        for i in range(rmol.n_atoms):
            atom_ptype = rmol.polarization_types[i]
            atype = rmol.assigned_atypes[i]
            index = -1

            for j, name in enumerate(ptype_names):
                if atom_ptype == name:
                    index = j
                    break
            assert index >= 0
            atype_map[atype] = index

        for pd in get_probe_dirs(rdir):
            all_found = True
            for req_fi in required_subdir_files:
                rf_path = join(pd, req_fi)
                if not (os.path.exists(rf_path) and os.path.isfile(rf_path)):
                    eprint(f"Directory {pd} lacks file {req_fi}: will not be included!")
                    all_found = False
                    break
            if all_found:
                pdirs_i.append(pd)
                target = np.genfromtxt(join(pd, 'MM_REF_BACK.pot'), usecols=pot_cols, skip_header=1, dtype=np.float64)
                target[:,3] += np.genfromtxt(join(pd, 'qm_polarization.pot'), usecols=[4], skip_header=1,
                                        dtype=np.float64)
                grid_points.append(target.shape[0])
                targets.append(target)
                pmol = StructureXYZ.StructXYZ(join(pd, 'QM_PR.xyz'), probe_types=probe_types)
                probe_mols.append(pmol)
                out_keys.append(version_file(pmol.key_file))
                # Will be many shallow copies.
                atypes_to_x_index.append(atype_map)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        opt_args = (probe_mols, out_keys, atypes_to_x_index, potential, targets, executor)
        bounds = (0, np.inf)
        eprint("Setup complete: beginning optimization.")
        ls_result = scipy.optimize.least_squares(cost_function, initial_x, jac='3-point', args=opt_args, verbose=2,
                                                 bounds=bounds, xtol=tol)
        eprint(f"Output of optimization: {ls_result}")
    """opt_args = (probe_mols, out_keys, atypes_to_x_index, potential, targets)
    bounds = (0, np.inf)
    eprint("Setup complete: beginning single-threaded optimization.")
    ls_result = scipy.optimize.least_squares(cost_function_sequential, initial_x, jac='3-point', args=opt_args, verbose=2,
                                             bounds=bounds, xtol=tol)
    eprint(f"Output of optimization: {ls_result}")"""


def get_molec_files(optinfo: str) -> Sequence[str]:
    files_out = []
    with open(optinfo, 'r') as r:
        for line in r:
            line = re.sub(r'#.+', '', line.rstrip())
            if len(line) > 0:
                if os.path.exists(line) and os.path.isfile(line):
                    files_out.append(line)
                else:
                    eprint(f"File ]{line}[ could not be found!")
    return files_out


def read_optinfo(readfile: str = None) -> (np.ndarray, np.ndarray):
    assert readfile is not None
    ptype_names = np.genfromtxt(readfile, usecols=[0], dtype=str)
    init_x = np.genfromtxt(readfile, usecols=[1], dtype=np.float64)
    return ptype_names, init_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tinkerpath', dest='tinker_path', type=str, default='', help='Path to Tinker '
                                                                                             'executables')
    parser.add_argument('-p', '--probetype', dest='probe_type', type=int, default=999, help='Probe atom type')
    parser.add_argument('-e', '--tol', dest='tol', type=float, default=None,
                        help='Cease optimization at this tolerance (otherwise algorithm-dependent default)')
    # parser.add_argument('-x', type=Sequence[float], default=None, help='Manually specify starting polarizabilities')
    parser.add_argument('-i', '--maxiter', dest='max_iter', type=int, default=250, help='Maximum iterations of the '
                                                                                        'optimizer')
    parser.add_argument('--opt_info', dest='optimize_info', type=str, default='molecules.txt',
                        help='File containing paths to molecules (i.e. path/to/QM_REF.xyz)')

    args = parser.parse_args()
    mfis = get_molec_files(args.optimize_info)
    assert len(mfis) > 0

    # TODO: Not hard-coded
    ptype_names, init_x = read_optinfo('polartype-define.txt')
    main_inner(mfis, ptype_names, init_x, tinker_path=args.tinker_path, probe_types=[args.probe_type], tol=args.tol,
               maxiter=args.max_iter)


if __name__ == "__main__":
    main()