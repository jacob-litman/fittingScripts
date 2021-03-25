import argparse
import re
import os
from typing import Sequence, Mapping

import JMLMath
from JMLUtils import eprint, get_probe_dirs, version_file
from os.path import join
import StructureXYZ
import numpy as np
import scipy.optimize
import time
import concurrent.futures
import sys
import PolarTypeReader
import EspAnalysis

DEFAULT_TOL = 1E-4
DEFAULT_MAX_ITER = 1000
DEFAULT_OUTFILE = 'out_polarizability.tsv'
required_subdir_files = ['qm_polarization.pot', 'MM_REF_BACK.pot', 'QM_PR.xyz', 'QM_PR.key']
pot_cols = [1, 2, 3, 4]
polarize_patt = re.compile(r'^(polarize\s+)(\d+\s+)(\d+\.\d+)(\s+.+)$')
molec_dir_patt = re.compile(r'^([0-9]{3})_(.+)_([^_]+)$')
probe_dir_patt = re.compile(r'^[A-Z]+[1-9][0-9]*')
DEFAULT_MPOL_WT = 0.001


def edit_keyfile(in_key: str, out_key: str, x: np.ndarray, atype_map: Mapping[int, int], submapping: Sequence[int],
                 precis: int = 12):
    with open(in_key, 'r') as r:
        with open(out_key, 'w') as w:
            for line in r:
                if line.startswith("polarize"):
                    m = polarize_patt.match(line)
                    assert m
                    atype = int(m.group(2).strip())
                    if atype in atype_map:
                        index = atype_map[atype] - 1
                        submap_index = submapping[index]
                        assert submap_index >= 0
                        out_polar = x[submap_index]
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
                  executor: concurrent.futures.Executor, submapping: Sequence[int], weights: np.ndarray = None,
                  sequential: bool = False, verbose: bool = False, qm_tensors: Sequence[np.ndarray] = None,
                  mpol_wt: float = DEFAULT_MPOL_WT) -> float:
    del_time = -time.time()
    tot_cost = 0
    costbundles = []

    if weights is None:
        weights = []
        for pml in probe_mols:
            wts = [1.0] * len(pml)
            weights.append(wts)

    for i, pml in enumerate(probe_mols):
        for j, pm in enumerate(pml):
            cbundle = (x, pm, out_keys[i][j], pt_maps[i], potential, targets[i][j], weights[i][j], submapping,
                       qm_tensors[i], mpol_wt)
            costbundles.append(cbundle)

    sys.stdout.flush()
    sys.stderr.flush()

    if sequential:
        for cb in costbundles:
            tot_cost += cost_interior(*cb)
    else:
        futures = {executor.submit(cost_interior, *costbundle) for costbundle in costbundles}
        for future in concurrent.futures.as_completed(futures):
            try:
                tot_cost += future.result()
            except Exception as exc:
                eprint(f"Exception in parallel processing: {exc}")
                raise exc

    if verbose:
        del_time += time.time()
        if sequential:
            eprint(f"Sequential evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
        else:
            eprint(f"Parallel evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
    sys.stdout.flush()
    sys.stderr.flush()
    return tot_cost


DEFAULT_MPOL_COMPONENT_WEIGHTS = np.array([1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)


def cost_interior(x: np.ndarray, pm: StructureXYZ.StructXYZ, outkey: str, atype_map: Mapping[int, int], potential: str,
                  target: np.ndarray, weight: float, submapping: Sequence[int], qm_tensor: np.ndarray,
                  mpol_wt: float, polarize: str = 'polarize') -> float:
    edit_keyfile(pm.key_file, outkey, x, atype_map, submapping)
    mm_potential = pm.get_esp(potential, outkey)
    assert np.array_equal(mm_potential[:, 0:2], target[:, 0:2])
    mm_potential[:, 3] -= target[:, 3]
    assert np.allclose(mm_potential[:, 0:2], target[:, 0:2])
    esp_component = np.average(np.square(mm_potential[:, 3]))

    mm_tensor = EspAnalysis.mm_polarization_tensor(pm.in_file, outkey, polarize, False)
    del_tensor = qm_tensor - mm_tensor
    del_tensor = np.sqrt(np.square(del_tensor) * DEFAULT_MPOL_COMPONENT_WEIGHTS)
    tensor_component = np.sum(del_tensor)

    return weight * (esp_component + (mpol_wt * tensor_component))


def write_cost(mol, keyfile: str, mm_ref_back = 'MM_REF_BACK.pot', qm_pol = 'qm_polarization.pot', qm_mpol_fi: str = None, potential: str = 'potential', polarize: str = 'polarize', write_grid_points: bool = False):
    """Write out components of the cost function"""
    if isinstance(mol, str):
        mol = StructureXYZ.StructXYZ(mol)
    assert isinstance(mol, StructureXYZ.StructXYZ)

    target = np.genfromtxt(mm_ref_back, usecols=pot_cols, skip_header=1, dtype=np.float64)
    eprint(f"Target shape: {target.shape}")
    target[:,3] += np.genfromtxt(qm_pol, usecols=[4], skip_header=1, dtype=np.float64)
    eprint(f"Updated target shape: {target.shape}")

    qm_tensor = EspAnalysis.qm_polarization_tensor(qm_mpol_fi)
    mm_tensor = EspAnalysis.mm_polarization_tensor(mol.in_file, keyfile, polarize, False)

    curr_esp = mol.get_esp(potential, keyfile)
    eprint(f"curr_esp shape: {curr_esp.shape}")
    assert np.allclose(curr_esp[:,0:2], target[:,0:2])
    diff1 = curr_esp[:, 3] - target[:, 3]
    diff2 = curr_esp[:,3] + target[:,3]
    d1_target = np.average(np.square(diff1))
    d2_target = np.average(np.square(diff2))
    assert d1_target < d2_target

    del_tensor = qm_tensor - mm_tensor
    eprint(f"ESP difference: {d1_target}\nTensor difference: {del_tensor}")


def summarize_mpol_diff(mpol_errs: np.ndarray) -> np.ndarray:
    summary = np.empty([4,7], dtype=np.float64)
    summary[0,:] = np.sqrt(np.mean(np.square(mpol_errs), axis=0))
    summary[1,:] = np.mean(mpol_errs, axis=0)
    summary[2,:] = np.mean(np.abs(mpol_errs), axis=0)
    summary[3,:] = np.std(mpol_errs, axis=0, ddof=1)
    return summary


def write_mpol_csv(outfi: str, vals: np.ndarray, molec_names: Sequence[str], ptype_ids: Sequence[Sequence[int]],
                   precis: int = 5):
    stats = summarize_mpol_diff(vals)
    stat_titles = ['RMSE', 'MSE', 'MUE', 'SD']
    assert precis < 15

    with open(outfi, 'w') as w:
        w.write("Molecule,Isotropic,xx,xy,yy,xz,yz,zz,Polar Type IDs\n")
        for i, mname in enumerate(molec_names):
            w.write(mname)
            for j in range(7):
                w.write(f",{vals[i][j]:.{precis}f}")
            for pt_id in ptype_ids[i]:
                w.write(f",{pt_id}")
            w.write('\n')
        w.write('\n')
        for i, st in enumerate(stat_titles):
            w.write(st)
            for j in range(7):
                w.write(f",{stats[i][j]:.{precis}f}")
            w.write('\n')
        w.write('\n')


DEFAULT_ORIG_ABS_FI = 'quantum_molpols.csv'
DEFAULT_ORIG_DIFF_FI = 'unfit_molpol_error.csv'
DEFAULT_REFIT_DIFF_FI = 'refit_molpol_error.csv'
DEFAULT_REL_ORIG_FI = 'unfit_molpol_relative.csv'
DEFAULT_REL_REFIT_FI = 'refit_molpol_relative.csv'
DEFAULT_QM_MOLPOL_FI = 'qm_molpols.csv'


def main_inner(tinker_path: str = '', ptypes_fi: str = 'polarTypes.tsv', n_threads: int=None,
               ref_files: Sequence[str] = None, tol: float = DEFAULT_TOL, sequential: bool = False,
               verbose: bool = False, outfile: str = DEFAULT_OUTFILE, qm_fi = DEFAULT_ORIG_ABS_FI,
               mpol_orig_fi = DEFAULT_ORIG_DIFF_FI, mpol_refit_fi = DEFAULT_REFIT_DIFF_FI,
               rel_orig_fi = DEFAULT_REL_ORIG_FI, rel_refit_fi = DEFAULT_REL_REFIT_FI,
               qm_mp_finame = DEFAULT_QM_MOLPOL_FI, mpol_wt: float = DEFAULT_MPOL_WT):

    ptyping = PolarTypeReader.PtypeReader(ptypes_fi)

    # Reference molecules: StructXYZ (QM_REF.xyz)
    ref_mols = []
    # List of Map[int, int] for atom type to polar type ID (1-indexed).
    polar_mappings = []
    # Number of probes for each molecule.
    nprobes = []
    # Set of all used polar types.
    ptypes_used = set()
    # List of tensors for QM molecular polarizability values.
    qm_molpols = []

    # 2D arrays
    # How much to weight each given probe (initially just 1 / no. of probes for this molecule)
    weights = []
    # Probe molecules: StructXYZ (QM_PR.xyz)
    probe_mols = []
    # Probe directories.
    pdirs = []
    # Probe delta-delta-ESP (qm_polarization.pot)
    targets = []
    # Where each probe starts in the flattened ESP array.
    esp_indices = []
    # Output .key files (QM_PR.key_0)
    out_keys = []
    # The (unique) polar types associated with each molecule.
    ptype_ids = []

    tot_grid_points = 0

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"
    polarize = tinker_path + "polarize"

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
        these_pts = [pt.id for pt in sorted(set(ptypes_i))]
        ptype_ids.append(these_pts)
        ptypes_used.update(these_pts)

        assert os.path.isfile(join(dir, qm_mp_finame))
        qm_mp = EspAnalysis.qm_polarization_tensor(join(dir, qm_mp_finame))
        qm_molpols.append(qm_mp)

        tgts = []
        pmols = []
        e_inds = []
        o_keys = []
        pdir_list = []

        num_probes = 0
        for sdire in os.scandir(dir):
            sdir = sdire.name
            to_sdir = join(dir, sdir)
            if not os.path.isdir(to_sdir) or not probe_dir_patt.match(sdir):
                continue

            target = np.genfromtxt(join(to_sdir, 'MM_REF_BACK.pot'), usecols=pot_cols, skip_header=1, dtype=np.float64)
            target[:,3] += np.genfromtxt(join(to_sdir, 'qm_polarization.pot'), usecols=[4], skip_header=1, dtype=np.float64)
            tgts.append(target)
            num_probes += 1

            pmols.append(StructureXYZ.StructXYZ(join(to_sdir, 'QM_PR.xyz')))
            e_inds.append(tot_grid_points)
            tot_grid_points += target.shape[0]
            o_keys.append(join(to_sdir, 'QM_PR.key_0'))
            pdir_list.append(sdir)

        targets.append(tgts)
        probe_mols.append(pmols)
        esp_indices.append(e_inds)
        nprobes.append(num_probes)
        out_keys.append(o_keys)
        pdirs.append(pdir_list)

        wt = 1.0 / num_probes
        wts = [wt] * num_probes
        weights.append(wts)

    # Following shenanigans are to eliminate unused polarizabilities from the fitting, eliminating needless target
    # evaluations.
    submapping = [-1 for pt in ptyping.ptypes]
    n_pt_used = 0
    initial_x = []

    for i in sorted(ptypes_used):
        submapping[i-1] = n_pt_used
        initial_x.append(ptyping.ptypes[i-1].initial_polarizability)
        #initial_x.append(ptyping.ptypes[submapping[i-1]].initial_polarizability)
        n_pt_used += 1

    eprint(f"submapping: {submapping}\ninitial X: {initial_x}\nptypes_used: {ptypes_used}")
    initial_x = np.array(initial_x, dtype=np.float64)

    lb = np.full_like(initial_x, 0.01)
    ub = initial_x * 4.0
    bounds = (lb, ub)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        opt_args = (probe_mols, out_keys, polar_mappings, potential, targets, executor, submapping, weights, sequential,
                    verbose, qm_molpols, mpol_wt)
        eprint("Setup complete: beginning optimization")
        ls_result = scipy.optimize.least_squares(cost_function, initial_x, jac='3-point', args=opt_args, verbose=2,
                                                 bounds=bounds, xtol=tol)
        eprint(f"Output of optimization: {ls_result}")
        x = ls_result.x

        with open(outfile, 'w') as w:
            w.write("ID\tSMARTS\tName\tFit Polarizabiliities\tInitial Polarizabilities\tDelta Polarizability\n")
            for i, ptype in enumerate(ptyping.ptypes):
                if submapping[i] >= 0:
                    ptype.polarizability = x[submapping[i]]
                    del_pol = ptype.polarizability - ptype.initial_polarizability
                    w.write(f"{ptype.id}\t{ptype.smarts_string}\t{ptype.name}\t{x[submapping[i]]:9.6f}"
                            f"\t{ptype.initial_polarizability:9.6f}\t{del_pol:9.6f}\n")

        qm_tensors = []
        orig_tensors = []
        refit_tensors = []
        molec_names = []

        for i, pmols in enumerate(probe_mols):
            for j, pmol in enumerate(pmols):
                pdir = join(ref_dirs[i], pdirs[i][j])
                write_cost(pmol, out_keys[i][j], join(pdir, 'MM_REF_BACK.pot'), join(pdir, 'qm_polarization.pot'), join(ref_dirs[i], qm_mp_finame), potential, polarize)
                '''write_cost(pmol, fit_key, join(pdir, 'MM_REF_BACK.pot'), join(pdir, 'qm_polarization.pot'),
                           join(rdir, qm_mp_finame), potential, polarize)'''
            '''qmr_xyz = join(rdir, 'QM_REF.xyz')
            orig_key = join(rdir, 'QM_REF.key')
            fit_key = out_keys[i][0]
            assert os.path.isfile(qmr_xyz) and os.path.isfile(orig_key) and os.path.isfile(fit_key)
            qm_tensor = qm_molpols[i]
            orig_tensor = EspAnalysis.mm_polarization_tensor(qmr_xyz, orig_key, polarize=polarize)
            refit_tensor = EspAnalysis.mm_polarization_tensor(qmr_xyz, fit_key, polarize=polarize)

            qm_tensors.append(qm_tensor)
            orig_tensors.append(orig_tensor)
            refit_tensors.append(refit_tensor)

            mname = os.path.basename(rdir)
            m = molec_dir_patt.match(mname.strip())
            assert m
            mname = m.group(3)
            molec_names.append(mname)

            pdir = join(rdir, pdirs[i][0])
            #qm_mpol_fi: str = None, potential: str = 'potential', polarize: str = 'polarize', write_grid_points: bool = False
            write_cost(ref_mols[i], fit_key, join(pdir, 'MM_REF_BACK.pot'), join(pdir, 'qm_polarization.pot'), join(rdir, qm_mp_finame), potential, polarize)'''

        qm_tensors = np.array(qm_tensors, dtype=np.float64)
        orig_tensors = np.array(orig_tensors, dtype=np.float64)
        refit_tensors = np.array(refit_tensors, dtype=np.float64)

        orig_diff = orig_tensors - qm_tensors
        refit_diff = refit_tensors - qm_tensors

        orig_rel = JMLMath.divide_ignore_zero(orig_diff, qm_tensors)
        refit_rel = JMLMath.divide_ignore_zero(refit_diff, qm_tensors)

        write_mpol_csv(qm_fi, qm_tensors, molec_names, ptype_ids)
        write_mpol_csv(mpol_orig_fi, orig_diff, molec_names, ptype_ids)
        write_mpol_csv(mpol_refit_fi, refit_diff, molec_names, ptype_ids)
        write_mpol_csv(rel_orig_fi, orig_rel, molec_names, ptype_ids)
        write_mpol_csv(rel_refit_fi, refit_rel, molec_names, ptype_ids)


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
    """parser.add_argument('-i', '--maxiter', dest='max_iter', type=int, default=250, help='Maximum iterations of the '
                                                                                        'optimizer')"""
    parser.add_argument('-n', '--nthreads', dest='n_threads', type=int, default=None,
                        help='Number of Tinker processes to run in parallel')
    parser.add_argument('--polar_types', dest='ptypes_fi', type=str, default='polarTypes.tsv',
                        help='Tab-separated file containing polarizability type information.')
    parser.add_argument('--opt_info', dest='optimize_info', type=str, default='molecules.txt',
                        help='File containing paths to molecules (i.e. path/to/QM_REF.xyz)')
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential',
                        help='Run potential commands sequentially (single-threaded: over-rides n_threads)')
    parser.add_argument('-o', '--output', dest='outfile', type=str, default=DEFAULT_OUTFILE,
                        help='File to write output (tab-separated values) to.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help='Print extra information (particularly timings for each target function evaluation).')
    parser.add_argument('-m', '--molpol_wt', dest='mpol_wt', type=float, default=DEFAULT_MPOL_WT,
                        help='Relative weighting of molecular polarizabilities vs. electrostatic potential.')

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

    main_inner(tinker_path=args.tinker_path, ptypes_fi=args.ptypes_fi, n_threads = args.n_threads, ref_files=ref_files,
               tol=args.tol, sequential=args.sequential, verbose=args.verbose, mpol_wt=args.mpol_wt)


if __name__ == "__main__":
    main()