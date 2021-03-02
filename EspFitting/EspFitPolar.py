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
import EspAnalysis

DEFAULT_TOL = 1E-4
DEFAULT_MAX_ITER = 1000
DEFAULT_OUTFILE = 'out_polarizability.tsv'
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
                  executor: concurrent.futures.Executor, weights: np.ndarray = None, sequential: bool = False) -> float:
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
            cbundle = (x, pm, out_keys[i][j], pt_maps[i], potential, targets[i][j], weights[i][j])
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

    del_time += time.time()
    if sequential:
        eprint(f"Sequential evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
    else:
        eprint(f"Parallel evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
    #eprint(f"Function evaluation in {del_time:.3f} sec: target value {tot_cost:.6g}")
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


def summarize_mpol_diff(mpol_errs: np.ndarray) -> np.ndarray:
    summary = np.empty([4,7], dtype=np.float64)
    summary[0,:] = np.sqrt(np.mean(np.square(mpol_errs), axis=0))
    summary[1,:] = np.mean(mpol_errs, axis=0)
    summary[2,:] = np.mean(np.abs(mpol_errs), axis=0)
    summary[3,:] = np.std(mpol_errs, axis=0, ddof=1)
    return summary


def main_inner(tinker_path: str = '', ptypes_fi: str = 'polarTypes.tsv', n_threads: int=None,
               ref_files: Sequence[str] = None, tol: float = DEFAULT_TOL, sequential: bool = False,
               outfile: str = DEFAULT_OUTFILE, mpol_orig_fi = 'unfit_molpol_error.csv',
               mpol_refit_fi = 'refit_molpol_error.csv'):
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

        tgts = []
        pmols = []
        e_inds = []
        o_keys = []

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

        targets.append(tgts)
        probe_mols.append(pmols)
        esp_indices.append(e_inds)
        nprobes.append(num_probes)
        out_keys.append(o_keys)

        wt = 1.0 / num_probes
        wts = [wt] * num_probes
        weights.append(wts)

    bounds = (0, np.inf)

    initial_x = [pt.initial_polarizability for pt in ptyping.ptypes]
    initial_x = np.array(initial_x, dtype=np.float64)

    #with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        opt_args = (probe_mols, out_keys, polar_mappings, potential, targets, executor, weights, sequential)
        eprint("Setup complete: beginning optimization")
        ls_result = scipy.optimize.least_squares(cost_function, initial_x, jac='3-point', args=opt_args, verbose=2,
                                                 bounds=bounds, xtol=tol)
        eprint(f"Output of optimization: {ls_result}")
        x = ls_result.x

        with open(outfile, 'w') as w:
            w.write("ID\tSMARTS\tName\tFit Polarizabiliities\tInitial Polarizabilities\tDelta Polarizability\n")
            for i, ptype in enumerate(ptyping.ptypes):
                ptype.polarizability = x[i]
                del_pol = ptype.polarizability - ptype.initial_polarizability
                w.write(f"{ptype.id}\t{ptype.smarts_string}\t{ptype.name}\t{x[i]:9.6f}"
                        f"\t{ptype.initial_polarizability:9.6f}\t{del_pol:9.6f}\n")

        qm_tensors = []
        orig_tensors = []
        refit_tensors = []
        molec_names = []

        for i, rdir in enumerate(ref_dirs):
            qmr_xyz = join(rdir, 'QM_REF.xyz')
            orig_key = join(rdir, 'QM_REF.key')
            #fit_key = join(rdir, 'QM_REF.key_0')
            fit_key = out_keys[i][0]
            eprint(f"{qmr_xyz} existence: {os.path.exists(qmr_xyz)}")
            eprint(f"{orig_key} existence: {os.path.exists(orig_key)}")
            eprint(f"{fit_key} existence: {os.path.exists(fit_key)}")
            assert os.path.isfile(qmr_xyz) and os.path.isfile(orig_key) and os.path.isfile(fit_key)
            qm_mp = join(rdir, 'qm_molpols.csv')
            if not os.path.isfile(qm_mp):
                eprint(f"Quantum mechanics molecular polarizability file {qm_mp} does not exist: skipping this "
                       f"molecule for molecular polarizability analysis.")

            qm_tensor = EspAnalysis.qm_polarization_tensor(qm_mp)
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

        qm_tensors = np.array(qm_tensors, dtype=np.float64)
        orig_tensors = np.array(orig_tensors, dtype=np.float64)
        refit_tensors = np.array(refit_tensors, dtype=np.float64)

        orig_diff = orig_tensors - qm_tensors
        refit_diff = refit_tensors - qm_tensors

        orig_stats = summarize_mpol_diff(orig_diff)
        refit_stats = summarize_mpol_diff(refit_diff)
        stat_titles = ['RMSE', 'MSE', 'MUE', 'SD']

        noln_epr_kw = {'end': ''}

        eprint("Original molecular polarizability errors\n\n")
        with open(mpol_orig_fi, 'w') as w:
            eprint(f"{'Molecule':<20s}{'Isotropic':>15s}{'xx':>15s}{'xy':>15s}{'yy':>15s}{'xz':>15s}{'yz':>15s}{'zz':>15s}")
            w.write("Molecule,Isotropic,xx,xy,yy,xz,yz,zz\n")
            for i, mname in enumerate(molec_names):
                eprint(f"{mname:<20s}", kwargs=noln_epr_kw)
                w.write(mname)
                for j in range(7):
                    eprint(f"   {orig_diff[i][j]:12.9f}", kwargs=noln_epr_kw)
                    w.write(f",{orig_diff[i][j]:12.9f}")
                eprint("")
                w.write('\n')
            for i, st in enumerate(stat_titles):
                eprint(f"{st:<20s}", kwargs=noln_epr_kw)
                w.write(st)
                for j in range(7):
                    eprint(f"   {orig_stats[i][j]:12.9f}", kwargs=noln_epr_kw)
                    w.write(f",{orig_stats[i][j]:12.9f}")
                eprint("")
                w.write('\n')


        eprint("\n\nRefit molecular polarizability errors\n\n")
        with open(mpol_refit_fi, 'w') as w:
            for i, mname in enumerate(molec_names):
                eprint(f"{mname:<20s}", kwargs=noln_epr_kw)
                w.write(mname)
                for j in range(7):
                    eprint(f"   {refit_diff[i][j]:12.9f}", kwargs=noln_epr_kw)
                    w.write(f",{orig_diff[i][j]:12.9f}")
                eprint("")
                w.write('\n')
            for i, st in enumerate(stat_titles):
                eprint(f"{st:<20s}", kwargs=noln_epr_kw)
                w.write(st)
                for j in range(7):
                    eprint(f"   {refit_stats[i][j]:12.9f}", kwargs=noln_epr_kw)
                    w.write(f",{refit_stats[i][j]:12.9f}")
                eprint("")
                w.write('\n')


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
               tol=args.tol, sequential=args.sequential)


if __name__ == "__main__":
    main()