#!/usr/bin/env python

import argparse
import os
import shutil
import re
from JMLUtils import eprint, verbose_call, name_to_atom
from typing import Sequence
import SubPots


def main_inner(tinker_path: str = '', gauss_path: str = '', probe_types: Sequence[int] = None):
    assert tinker_path is not None and gauss_path is not None
    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

    if probe_types is None:
        probe_types = [999]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    if not gauss_path.endswith(os.sep) and gauss_path != '':
        gauss_path += os.sep
    formchk = gauss_path + "formchk"
    cubegen = gauss_path + "cubegen"
    #eprint(f"Command paths: potential is {potential}\nformchk is {formchk}\ncubegen is {cubegen}\n")
    verbose_call([formchk, 'QM_REF.chk'])

    for pdir in probe_dirs:
        shutil.copy2('QM_REF.fchk', pdir)
        os.chdir(pdir)
        at_name = re.sub(r"^\./", '', pdir)
        foc_atom = name_to_atom('QM_PR.xyz', at_name)
        eprint(f"Operating in directory {pdir}, atom {foc_atom[1]}{foc_atom[0]}\n")

        # Writes the grid file for cubegen
        verbose_call([potential, "1", "QM_PR.xyz", 'QM_PR.key'])
        # Formats the binary .chk to .fchk for cubegen
        verbose_call([formchk, "QM_PR.chk"])

        # TODO: Check that it actually is MP2 potential.
        # Call cubegen to write out the QM potential to .cube file.
        with open('QM_PR.grid', 'r') as f:
            verbose_call([cubegen, '0', 'potential=MP2', 'QM_PR.fchk', 'QM_PR.cube', '-5', 'h'], kwargs={'stdin': f})

        # Convert .cube to .pot
        verbose_call([potential, "2", "QM_PR.cube"])
        # Write out PR_NREF.pot (probe charge only potential)
        verbose_call([potential, "3", "PR_NREF.xyz", "Y"])

        # Add the probe background (PR_NREF.pot) to the QM potential to get QM-with-probe potential.
        with open('QM_PR_BACK.pot', 'w') as f:
            with open('QM_PR_BACK.potlog', 'w') as f2:
                SubPots.main_inner('QM_PR.pot', 'PR_NREF.pot', out=f, err=f2, subtract=False,
                                   header_comment='QM_PR_BACK.pot')

        # Write out how different the MM potential is from the QM potential (with probe).
        with open('unfit_diff.log', 'w') as f:
            verbose_call([potential, "5", "QM_PR.xyz", "QM_PR_BACK.pot", "Y"], kwargs={'stdout': f})

        # Using the with-probe grid, write out the QM reference potential (no probe).
        with open('QM_PR.grid', 'r') as f:
            verbose_call([cubegen, '0', 'potential=MP2', 'QM_REF.fchk', 'QM_REF.cube', '-5', 'h'], kwargs={'stdin': f})
        # Convert .cube to .pot.
        verbose_call([potential, '2', 'QM_REF.cube'])

        # Calculate the effect of polarization and write to .pot (QM).
        with open('qm_polarization.pot', 'w') as f:
            with open('polarize_diff_qm.log', 'w') as f2:
                SubPots.main_inner('QM_PR.pot', 'QM_REF.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                   y=foc_atom[3], z=foc_atom[4], header_comment='qm_polarization.pot')

        # Not included in the original eval scripts.
        # Possible OS incompatibility
        os.symlink('QM_PR.xyz', 'MM_PR.xyz')
        shutil.copy2('QM_PR.key', 'MM_PR.key')
        verbose_call([potential, '3', 'MM_PR.xyz', 'Y'])
        with open('mm_polarization.pot', 'w') as f:
            with open('polarize_diff_mm.log', 'w') as f2:
                SubPots.main_inner('MM_PR.pot', 'PR_NREF.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                   y=foc_atom[3], z=foc_atom[4], header_comment='mm_polarization.pot')
        os.symlink('QM_PR.xyz', 'temp.xyz')
        with open('QM_PR.key', 'r') as r:
            with open('temp.key', 'w') as w:
                for line in r:
                    if line.startswith('multipole'):
                        out = line.rstrip()
                        if int(out.split()[1]) in probe_types:
                            out = re.sub(r'-?\d+\.\d+ *$', '0.00000', out)
                            w.write(f"{out}\n")
                        else:
                            w.write(line)
                        pass
                    else:
                        w.write(line)
        eprint("Generating MM_REF.pot using temp.xyz and temp.key")
        verbose_call([potential, '3', 'temp.xyz', '-k', 'temp.key', 'Y'])
        shutil.move('temp.pot', 'MM_REF.pot')
        os.remove('temp.xyz')
        os.remove('temp.key')
        eprint('\n')
        os.chdir("..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='probe_type', type=int, default=999, help='Probe atom type')
    parser.add_argument('-t', dest='tinker_path', type=str, default='', help='Path to Tinker executables')
    parser.add_argument('-g', dest='gauss_path', type=str, default='', help='Path to Gaussian executables')
    args = parser.parse_args()
    main_inner(tinker_path=args.tinker_path, gauss_path=args.gauss_path, probe_types=[args.probe_type])


if __name__ == "__main__":
    main()
