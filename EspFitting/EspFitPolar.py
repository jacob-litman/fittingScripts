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
import PolarTypeReader

DEFAULT_TOL = 1E-4
DEFAULT_MAX_ITER = 1000
required_subdir_files = ['qm_polarization.pot', 'MM_REF_BACK.pot', 'QM_PR.xyz', 'QM_PR.key']
pot_cols = [1, 2, 3, 4]
polarize_patt = re.compile(r'^(polarize\s+)(\d+\s+)(\d+\.\d+)(\s+.+)$')
molec_dir_patt = re.compile(r'^([0-9]{3})_(.+)_([^_]+)$')
probe_dir_patt = re.compile(r'^[A-Z]+[1-9][0-9]*')


def edit_keyfile(in_key: str, out_key: str, x: np.ndarray, atype_map: Mapping[int, int], precis: int = 12):
    with open(in_key, 'r') as r:
        with open(out_key, 'w') as w:
            for line in r:
                if line.startswith("polarize"):
                    m = polarize_patt.match(line)
                    assert m
                    atype = int(m.group(2).strip())
                    if atype in atype_map:
                        out_polar = x[atype_map[atype] - 1]
                        w.write(f"{m.group(1)}{m.group(2)}{out_polar:.{precis}f}{m.group(4).rstrip()}\n")
                    else:
                        w.write(line)
                else:
                    w.write(line)


def check_finished_qm(dirn: str) -> bool:
    ref_found = False
    probe_found = False
    for root, dirs, files in os.walk(dirn):
        if 'QM_REF.psi4.dat' in files or 'QM_REF.log' in files:
            ref_found = True
        if 'QM_PR.psi4.dat' in files or 'QM_PR.log' in files:
            probe_found = True
    return ref_found and probe_found


def cost_function(x: np.ndarray, probe_mols: Sequence[Sequence[StructureXYZ.StructXYZ]], out_keys: Sequence[str],
                  pt_maps: Sequence[Mapping[int, int]], potential: str, targets: Sequence[Sequence[np.ndarray]],
                  executor: concurrent.futures.Executor, weights: np.ndarray = None) -> float:
    del_time = -time.time()
    tot_cost = 0
    costbundles = []

    if weights is None:
        weights = np.ones_like(probe_mols)

    for i, pml in enumerate(probe_mols):
        for j, pm in enumerate(pml):
            costbundles.append((x, pm, out_keys[i][j], pt_maps[i], potential, targets[i][j], weights[i][j]))

    futures = {executor.submit(cost_interior, *costbundle) for costbundle in costbundles}
    for future in concurrent.futures.as_completed(futures):
        tot_cost += future.result()

    del_time += time.time()
    eprint(f"Function evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
    sys.stdout.flush()
    sys.stderr.flush()
    return tot_cost


def cost_interior(x: np.ndarray, pm: StructureXYZ.StructXYZ, outkey: str, atype_map: Mapping[int, int], potential: str,
                  target: np.ndarray, weight: float) -> float:
    edit_keyfile(pm.key_file, outkey, x, atype_map)
    mm_potential = pm.get_esp(potential, outkey)
    assert np.array_equal(mm_potential[:, 0:2], target[:, 0:2])
    mm_potential[:, 3] -= target[:, 3]
    return weight * np.average(np.square(mm_potential[:, 3]))


def main_inner(tinker_path: str = '', ptypes_fi: str = 'polarTypes.tsv', n_threads: int=None,
               ref_files: Sequence[str] = None, maxiter: int = DEFAULT_MAX_ITER):
    ptyping = PolarTypeReader.PtypeReader(ptypes_fi)

    # Reference molecules: StructXYZ (QM_REF.xyz)
    ref_mols = []
    # List of Map[int, int] for atom type to polar type ID (1-indexed).
    polar_mappings = []
    # Number of probes for each molecule.
    nprobes = []

    # 2D arrays
    # How much to weight each given probe (initially just 1 / no. of probes for this molecule)
    weights = []
    # Probe molecules: StructXYZ (QM_PR.xyz)
    probe_mols = []
    # Probe delta-delta-ESP (qm_polarization.pot)
    targets = []
    # Where each probe starts in the flattened ESP array.
    esp_indices = []
    # Output .key files (QM_PR.key_0)
    out_keys = []

    tot_grid_points = 0


    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    if ref_files is None:
        ref_dirs = []
        for dir in os.scandir('.'):
            dirn = dir.name
            if not os.path.isdir(dirn):
                continue
            m = molec_dir_patt.match(dirn)
            if not m:
                continue
            if not check_finished_qm(dirn):
                continue
            ref_dirs.append(dirn)
    else:
        ref_dirs = [os.path.dirname(f) for f in ref_files]

    for dir in ref_dirs:
        qmr = join(dir, 'QM_REF.xyz')
        sqmr = StructureXYZ.StructXYZ(qmr)
        ref_mols.append(sqmr)

        ptypes_i, mapping_i, pt2 = PolarTypeReader.main_inner(sqmr, False, ptyping=ptyping)
        assert pt2 == ptyping
        polar_mappings.append(mapping_i)

        tgts = []
        pmols = []
        e_inds = []
        o_keys = []

        num_probes = 0
        for sdire in os.scandir(dir):
            sdir = sdire.name
            if not os.path.isdir(sdir) or not probe_dir_patt.match(sdir):
                continue

            to_sdir = join(dir, sdir)
            target = np.genfromtxt(join(to_sdir, 'MM_REF_BACK.pot'), usecols=pot_cols, skip_header=1, dtype=np.float64)
            target[:,3] += np.genfromtxt(join(to_sdir, 'qm_polarization.pot'), usecols=[4], skip_header=1, dtype=np.float64)
            tgts.append(target)
            num_probes += 1

            pmols.append(join(to_sdir, 'QM_PR.xyz'))
            e_inds.append(tot_grid_points)
            tot_grid_points += target.shape[0]
            o_keys.append(join(to_sdir, 'QM_PR.key_0'))

        targets.append(tgts)
        probe_mols.append(pmols)
        esp_indices.append(e_inds)
        nprobes.append(num_probes)
        out_keys.append(o_keys)

        wt = 1.0 / num_probes
        weights.append([wt] * num_probes)

        bounds = (0, np.inf)
        tol = DEFAULT_TOL

    initial_x = np.ndarray([pt.initial_polarizability for pt in ptyping.ptypes], dtype=np.float64)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        opt_args = (probe_mols, out_keys, polar_mappings, potential, targets, executor)
        eprint("Setup complete: beginning optimization")
        ls_result = scipy.optimize.least_squares(cost_function, initial_x, jac='3-point', args=opt_args, verbose=2, bounds=bounds, xtol=tol)
        eprint(f"Output of optimization: {ls_result}")


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
    parser.add_argument('-i', '--maxiter', dest='max_iter', type=int, default=250, help='Maximum iterations of the '
                                                                                        'optimizer')
    parser.add_argument('-n', '--nthreads', dest='n_threads', type=int, default=None,
                        help='Number of Tinker processes to run in parallel')
    parser.add_argument('--polar_types', dest='ptypes_fi', type=str, default='polarTypes.tsv',
                        help='Tab-separated file containing polarizability type information.')
    parser.add_argument('--opt_info', dest='optimize_info', type=str, default='molecules.txt',
                        help='File containing paths to molecules (i.e. path/to/QM_REF.xyz)')

    args = parser.parse_args()
    mfis = get_molec_files(args.optimize_info)
    assert len(mfis) > 0

    if args.optimize_info is not None:
        ref_files = []
        with open(args.optimize_info, 'r') as r:
            for line in r:
                line = re.sub(r'#.+', '', line.strip()).strip()
                if line != "" and os.path.isfile(line):
                    ref_files.append(line)
    else:
        ref_files = None

    main_inner(tinker_path=args.tinker_path, ptypes_fi=args.ptypes_fi, n_threads = args.n_threads, ref_files=ref_files, maxiter=args.max_iter)


if __name__ == "__main__":
    main()