#!/usr/bin/env python

import argparse
import os
import shutil
import re
from JMLUtils import eprint, verbose_call, name_to_atom
from OptionParser import OptParser
from typing import Sequence, FrozenSet
from Psi4GridToPot import psi4_grid2pot
import SubPots

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


def main_inner(opts: OptParser, tinker_path: str = '', gauss_path: str = '', probe_types: Sequence[int] = None):
    assert tinker_path is not None and gauss_path is not None
    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

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

    if not gauss_path.endswith(os.sep) and gauss_path != '':
        gauss_path += os.sep
    formchk = gauss_path + "formchk"
    cubegen = gauss_path + "cubegen"

    if not is_psi4:
        verbose_call([formchk, 'QM_REF.chk'])

    for pdir in probe_dirs:
        shutil.copy2('QM_REF.fchk', pdir)
        os.chdir(pdir)
        at_name = re.sub(r"^\./", '', pdir)
        foc_atom = name_to_atom('QM_PR.xyz', at_name)
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

        # Write out PR_NREF.pot (probe charge only potential)
        verbose_call([potential, "3", "PR_NREF.xyz", "Y"])

        # Add the probe background (PR_NREF.pot) to the QM potential to get QM-with-probe potential.
        eprint("Adding QM_PR.pot to PR_NREF.pot to generate QM_PR_BACK.pot")
        with open('QM_PR_BACK.pot', 'w') as f:
            with open('QM_PR_BACK.pot.log', 'w') as f2:
                SubPots.main_inner('QM_PR.pot', 'PR_NREF.pot', out=f, err=f2, subtract=False,
                                   header_comment='QM_PR_BACK.pot')

        # Write out how different the MM potential is from the QM potential (with probe).
        with open('unfit_diff.log', 'w') as f:
            verbose_call([potential, "5", "QM_PR.xyz", "QM_PR_BACK.pot", "Y"], kwargs={'stdout': f})

        # Using the with-probe grid, write out the QM reference potential (no probe).
        if is_psi4:
            psi4_grid2pot('QM_REF.pot', method=qm_method, esp='QM_REF.grid_esp.dat')
        else:
            with open('QM_PR.grid', 'r') as f:
                verbose_call([cubegen, '0', 'potential=MP2', 'QM_REF.fchk', 'QM_REF.cube', '-5', 'h'], kwargs={'stdin': f})
            # Convert .cube to .pot.
            verbose_call([potential, '2', 'QM_REF.cube'])

        # Calculate the effect of polarization and write to .pot (QM).
        eprint("Writing out QM polarization of ESP to qm_polarization.pot")
        with open('qm_polarization.pot', 'w') as f:
            with open('polarize_diff_qm.log', 'w') as f2:
                SubPots.main_inner('QM_PR.pot', 'QM_REF.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                   y=foc_atom[3], z=foc_atom[4], header_comment='qm_polarization.pot')

        # Not included in the original eval scripts.
        # Possible OS incompatibility
        os.symlink('QM_PR.xyz', 'MM_PR.xyz')
        os.symlink('QM_PR.key', 'MM_PR.key')
        verbose_call([potential, '3', 'MM_PR.xyz', 'Y'])

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
        verbose_call([potential, '3', 'MM_REF.xyz', 'Y'])

        eprint("Generating MM_REF_BACK.pot (MM_REF with probe background added back in)")
        with open('MM_REF_BACK.pot', 'w') as f:
            with open('mm_add_background.log', 'w') as f2:
                SubPots.main_inner('MM_REF.pot', 'PR_NREF.pot', out=f, err=f2, subtract=False,
                                   header_comment='MM_REF_BACK.pot')

        eprint("Writing out preliminary MM polarization of ESP to mm_polarization.pot")
        with open('mm_polarization.pot', 'w') as f:
            with open('polarize_diff_mm.log', 'w') as f2:
                SubPots.main_inner('MM_PR.pot', 'MM_REF_BACK.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                   y=foc_atom[3], z=foc_atom[4], header_comment='mm_polarization.pot')


        eprint("Writing out delta-delta ESP (QM polarization - MM polarization) to ddesp_qm_mm.pot")
        with open('ddesp_qm_mm.pot', 'w') as f:
            with open('ddesp_qm_mm.pot.log', 'w') as f2:
                SubPots.main_inner('qm_polarization.pot', 'mm_polarization.pot', out=f, err=f2, subtract=True,
                                   x=foc_atom[2], y=foc_atom[3], z=foc_atom[4], header_comment='ddesp_qm_mm.pot')

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
