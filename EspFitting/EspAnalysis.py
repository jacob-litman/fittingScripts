#!/usr/bin/env python

import argparse
import os
import re
import shutil
import sys
from typing import Sequence, FrozenSet
import subprocess

import numpy as np

import JMLUtils
import SubPots
from JMLUtils import eprint, verbose_call, name_to_atom, get_probe_dirs
from OptionParser import OptParser
from Psi4GridToPot import psi4_grid2pot

mpolar_patt = re.compile(r'^((?:multipole|polarize)\s+\d+\s+)\d+\.\d+(\s.+)?$')
scr_patt = re.compile(r'^ *psi4_io.set_default_path\(["\'](.+)["\']\) *$')
# Files which the script expects to be present in CWD.
dir_files_psi4 = frozenset(('PR_NREF.key', 'QM_PR.key', 'QM_REF.psi4', 'grid_esp.dat', 'grid.dat'))
dir_files_gauss = frozenset(('PR_NREF.key', 'QM_PR.key', 'QM_REF.com', 'QM_REF.chk'))
# Files which the script expects to be present in all probe subdirectories.
subdir_files_psi4 = frozenset(('PR_NREF.key', 'PR_NREF.xyz', 'QM_PR.psi4', 'QM_PR.key', 'QM_PR.xyz', 'QM_REF.key',
                               'grid.dat', 'QM_PR.grid_esp.dat', 'QM_REF.grid_esp.dat'))
subdir_files_gauss = frozenset(('PR_NREF.key', 'PR_NREF.xyz', 'QM_PR.chk', 'QM_PR.com', 'QM_PR.key', 'QM_PR.xyz',
                          'QM_REF.key'))


def check_files_present(probe_dirs: Sequence[str], dir_files: FrozenSet[str], subdir_files: FrozenSet[str]) -> Sequence[str]:
    curr_files = [os.path.basename(f.name) for f in os.scandir('.')]
    missing_files = []
    for fi in dir_files:
        if fi not in curr_files:
            missing_files.append(fi)
    for pd in probe_dirs:
        curr_files = [os.path.basename(f.name) for f in os.scandir(pd)]
        for fi in subdir_files:
            if fi not in curr_files:
                missing_files.append(f"{pd}{os.sep}{fi}")
    return missing_files


tensor_elements = {"iso", "aniso", "xx", "yx", "yy", "zx", "zy", "zz"}
default_mpol_out = "qm_molpols.csv"


def parse_molpols(mpol_log: str = "molpols.log", mpol_out: str = default_mpol_out) -> bool:
    if not os.path.exists(mpol_log):
        eprint(f"Did not find log file {mpol_log}")
        return False
    with open(mpol_log, 'r') as r:
        polarizes_found = False
        polarizabilities = []

        for line in r:
            line = line.strip()
            if line == "Dipole polarizability, Alpha (input orientation).":
                polarizes_found = True
            elif polarizes_found:
                toks = line.split()
                if toks[0] in tensor_elements:
                    polarizability = JMLUtils.parse_gauss_float(toks[1])
                    polarizability *= JMLUtils.gauss_polar_convert
                    polarizabilities.append(polarizability)
                    if toks[0] == "zz":
                        polarizes_found = False

        if len(polarizabilities) == 8:
            with open(mpol_out, 'w') as w:
                w.write("Isotropic,Anisotropic,xx,xy,yy,xz,yz,zz\n")
                for i, p in enumerate(polarizabilities):
                    if i == 0:
                        w.write(str(p))
                    elif i == 7:
                        w.write(f",{p}\n")
                    else:
                        w.write(f",{p}")
            return True
        else:
            eprint(f"Log file {mpol_log} did not contain polarizabilities!")
            return False


def compare_molpols(input_xyz: str, input_key: str, polarize: str = 'polarize', qm_mpol: str = default_mpol_out):
    if not os.path.exists(qm_mpol) or not os.path.isfile(qm_mpol):
        eprint(f"Molecular polarizabilities file {qm_mpol} does not exist or is not a file.")
        return
    assert input_xyz is not None and input_key is not None

    eprint(f"Calling (with input capture): {polarize} {input_xyz} {input_key}")
    output = subprocess.check_output([polarize, input_xyz, input_key])

    found_tensor = False
    tensor_lines = []
    isotropic_pol = -1
    for line in output.splitlines():
        line = str(line, encoding=sys.getdefaultencoding()).strip()
        if line.startswith("Molecular Polarizability Tensor"):
            found_tensor = True
        elif found_tensor:
            if line.startswith("Polarizability Tensor Eigenvalues"):
                found_tensor = False
            toks = line.split()
            if len(toks) == 3:
                tensor_lines.append(toks)
        elif line.startswith("Interactive Molecular Polarizability"):
            toks = line.split()
            isotropic_pol = float(toks[-1])

    assert len(tensor_lines) == 3 and isotropic_pol >= 0
    # Isotropic, xx, xy, yy, xz, yz, zz
    mm_pols = [isotropic_pol, float(tensor_lines[0][0]), float(tensor_lines[0][1]), float(tensor_lines[1][1]),
              float(tensor_lines[0][2]), float(tensor_lines[1][2]), float(tensor_lines[2][2])]
    mm_tensor = np.array(mm_pols, dtype=np.float64)

    # Skip the anisotropic element of the QM polarizabilities.
    qm_tensor = np.genfromtxt(qm_mpol, skip_header=1, usecols=[0, 2, 3, 4, 5, 6, 7], delimiter=',', dtype=np.float64)

    eprint(f"MM tensor: {mm_tensor}\nQM tensor: {qm_tensor}")
    diff_tensor = mm_tensor - qm_tensor
    eprint(f"Difference (MM - QM): {diff_tensor}")


def main_inner(opts: OptParser, tinker_path: str = '', gauss_path: str = '', probe_types: Sequence[int] = None):
    assert tinker_path is not None and gauss_path is not None
    probe_dirs = get_probe_dirs()

    if opts.is_psi4():
        is_psi4 = True
        dir_files = dir_files_psi4
        subdir_files = subdir_files_psi4
    else:
        is_psi4 = False
        dir_files = dir_files_gauss
        subdir_files = subdir_files_gauss

    qm_pot = opts.gauss_potential_type()
    qm_method = f"{opts.options['espmethod']}/{opts.options['espbasisset']}"

    missing_files = check_files_present(probe_dirs, dir_files, subdir_files)
    if len(missing_files) > 0:
        err_msg = "Error: required files not found:"
        for mf in missing_files:
            err_msg += f"\n{mf}"
        raise FileNotFoundError(err_msg)

    if probe_types is None:
        probe_types = [999]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"
    polarize = os.path.join(tinker_path, "polarize")

    if not gauss_path.endswith(os.sep) and gauss_path != '':
        gauss_path += os.sep
    formchk = gauss_path + "formchk"
    cubegen = gauss_path + "cubegen"

    if not is_psi4:
        verbose_call([formchk, 'QM_REF.chk'])

    parse_molpols()
    compare_molpols('QM_REF.xyz', 'QM_REF.key', polarize=polarize)

    for pdir in probe_dirs:
        shutil.copy2('QM_REF.fchk', pdir)
        os.chdir(pdir)
        at_name = re.sub(r"^\./", '', pdir)
        foc_atom = name_to_atom('QM_PR.xyz', at_name)
        if foc_atom is None:
            focus = False
            eprint(f"Operating in directory {pdir} with no specific atom\n")
            pass
        else:
            focus = True
            eprint(f"Operating in directory {pdir}, atom {foc_atom[1]}{foc_atom[0]}\n")
        if is_psi4:
            esp_fi = 'QM_PR.grid_esp.dat'
        else:
            esp_fi = None
            # Writes the grid file for cubegen
            verbose_call([potential, "1", "QM_PR.xyz", 'QM_PR.key'])

        # With Gaussian: format .chk to .fchk and run cubegen to get the QM ESP.
        # With Psi4: assert that both have already been done.
        # TODO: Support other QM packages.
        if not is_psi4:
            verbose_call([formchk, "QM_PR.chk"])
            with open('QM_PR.grid', 'r') as f:
                verbose_call([cubegen, '0', f'potential={qm_pot}', 'QM_PR.fchk', 'QM_PR.cube', '-5', 'h'], kwargs={'stdin': f})
            # Convert .cube to .pot
            verbose_call([potential, "2", "QM_PR.cube"])
        else:
            eprint(f"Combining {esp_fi} and grid.dat into QM_PR.pot")
            psi4_grid2pot('QM_PR.pot', method=qm_method, esp=esp_fi)

        # Symlink the grid file.
        JMLUtils.symlink_nofail("QM_PR.grid", "PR_NREF.grid")
        # Write out PR_NREF.pot (probe charge only potential)
        verbose_call([potential, "3", "PR_NREF.xyz", "PR_NREF.key", "Y", "PR_NREF.key"])

        # Add the probe background (PR_NREF.pot) to the QM potential to get QM-with-probe potential.
        eprint("Adding QM_PR.pot to PR_NREF.pot to generate QM_PR_BACK.pot")
        with open('QM_PR_BACK.pot', 'w') as f:
            with open('QM_PR_BACK.pot.log', 'w') as f2:
                SubPots.main_inner('QM_PR.pot', 'PR_NREF.pot', out=f, err=f2, subtract=False,
                                   header_comment='QM_PR_BACK.pot')

        # Write out how different the MM potential is from the QM potential (with probe).
        with open('unfit_diff.log', 'w') as f:
            verbose_call([potential, "5", "QM_PR.xyz", "QM_PR.key", "QM_PR_BACK.pot", "Y", "QM_PR.key"],
                         kwargs={'stdout': f})

        # Using the with-probe grid, write out the QM reference potential (no probe).
        if is_psi4:
            psi4_grid2pot('QM_REF.pot', method=qm_method, esp='QM_REF.grid_esp.dat')
        else:
            with open('QM_PR.grid', 'r') as f:
                verbose_call([cubegen, '0', f'potential={qm_pot}', 'QM_REF.fchk', 'QM_REF.cube', '-5', 'h'], kwargs={'stdin': f})
            # Convert .cube to .pot.
            verbose_call([potential, '2', 'QM_REF.cube'])

        # Calculate the effect of polarization and write to .pot (QM).
        eprint("Writing out QM polarization of ESP to qm_polarization.pot")
        with open('qm_polarization.pot', 'w') as f:
            with open('polarize_diff_qm.log', 'w') as f2:
                if focus:
                    SubPots.main_inner('QM_PR.pot', 'QM_REF.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                       y=foc_atom[3], z=foc_atom[4], header_comment='qm_polarization.pot')
                else:
                    SubPots.main_inner('QM_PR.pot', 'QM_REF.pot', out=f, err=f2, subtract=True,
                                       header_comment='qm_polarization.pot')

        # Not included in the original eval scripts.
        # Possible OS incompatibility
        os.symlink('QM_PR.xyz', 'MM_PR.xyz')
        os.symlink('QM_PR.key', 'MM_PR.key')
        verbose_call([potential, '3', 'MM_PR.xyz', 'MM_PR.key', 'Y', 'MM_PR.key'])

        eprint("Generating MM_REF.pot")
        os.symlink('QM_PR.xyz', 'MM_REF.xyz')
        with open('QM_PR.key', 'r') as r:
            with open('MM_REF.key', 'w') as w:
                for line in r:
                    if line.startswith('multipole') or line.startswith('polarize'):
                        atype = int(line.split()[1])
                        if atype in probe_types:
                            m = mpolar_patt.match(line)
                            if not m:
                                eprint("Failure to match: " + line)
                            assert m
                            w.write(f"{m.group(1)}0.00000")
                            if m.group(2) is not None:
                                w.write(m.group(2))
                            w.write("\n")

                        else:
                            w.write(line)
                    else:
                        w.write(line)
        verbose_call([potential, '3', 'MM_REF.xyz', 'MM_REF.key', 'Y', 'MM_REF.key'])

        eprint("Generating MM_REF_BACK.pot (MM_REF with probe background added back in)")
        with open('MM_REF_BACK.pot', 'w') as f:
            with open('mm_add_background.log', 'w') as f2:
                SubPots.main_inner('MM_REF.pot', 'PR_NREF.pot', out=f, err=f2, subtract=False,
                                   header_comment='MM_REF_BACK.pot')

        eprint("Writing out preliminary MM polarization of ESP to mm_polarization.pot")
        with open('mm_polarization.pot', 'w') as f:
            with open('polarize_diff_mm.log', 'w') as f2:
                if focus:
                    SubPots.main_inner('MM_PR.pot', 'MM_REF_BACK.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                       y=foc_atom[3], z=foc_atom[4], header_comment='mm_polarization.pot')
                else:

                    SubPots.main_inner('MM_PR.pot', 'MM_REF_BACK.pot', out=f, err=f2, subtract=True,
                                       header_comment='mm_polarization.pot')

        eprint("Writing out delta-delta ESP (QM polarization - MM polarization) to ddesp_qm_mm.pot")
        with open('ddesp_qm_mm.pot', 'w') as f:
            with open('ddesp_qm_mm.pot.log', 'w') as f2:
                if focus:
                    SubPots.main_inner('qm_polarization.pot', 'mm_polarization.pot', out=f, err=f2, subtract=True,
                                       x=foc_atom[2], y=foc_atom[3], z=foc_atom[4], header_comment='ddesp_qm_mm.pot')
                else:
                    SubPots.main_inner('qm_polarization.pot', 'mm_polarization.pot', out=f, err=f2, subtract=True,
                                       header_comment='ddesp_qm_mm.pot')

        eprint('\n')
        os.chdir("..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='probe_type', type=int, default=999, help='Probe atom type')
    parser.add_argument('-t', dest='tinker_path', type=str, default='', help='Path to Tinker executables')
    parser.add_argument('-g', dest='gauss_path', type=str, default='', help='Path to Gaussian executables')
    parser.add_argument('-o', dest='opts_file', type=str, default=None,
                        help='File containing key=value properties: default locations: poltype.ini, espfit.ini')
    args = parser.parse_args()
    opts = OptParser(args.opts_file)
    main_inner(opts, tinker_path=args.tinker_path, gauss_path=args.gauss_path, probe_types=[args.probe_type])


if __name__ == "__main__":
    main()
