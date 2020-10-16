#!/usr/bin/env python

import argparse
import os
import shutil
import re
from JMLUtils import eprint, verbose_call, name_to_atom
import SubPots


def main_inner(tinker_path: str = '', gauss_path: str = ''):
    assert tinker_path is not None and gauss_path is not None
    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + "potential"

    if not gauss_path.endswith(os.sep) and gauss_path != '':
        gauss_path += os.sep
    formchk = gauss_path + "formchk"
    cubegen = gauss_path + "cubegen"\
    #eprint(f"Command paths: potential is {potential}\nformchk is {formchk}\ncubegen is {cubegen}\n")
    verbose_call([formchk, 'QM_REF.chk'])

    for pdir in probe_dirs:
        shutil.copy2('QM_REF.fchk', pdir)
        os.chdir(pdir)
        at_name = re.sub(r"^\./", '', pdir)
        foc_atom = name_to_atom('QM_PR.xyz', at_name)
        eprint(f"Operating in directory {pdir}, atom {foc_atom[1]}{foc_atom[0]}\n")

        verbose_call([potential, "1", "QM_PR.xyz", 'QM_PR.key'])
        verbose_call([formchk, "QM_PR.chk"])

        # TODO: Check that it actually is MP2 potential.
        with open('QM_PR.grid', 'r') as f:
            verbose_call([cubegen, '0', 'potential=MP2', 'QM_PR.fchk', 'QM_PR.cube', '-5', 'h'], kwargs={'stdin': f})

        verbose_call([potential, "2", "QM_PR.cube"])
        verbose_call([potential, "3", "PR_NREF.xyz", "Y"])

        with open('QM_PR_BACK.pot', 'w') as f:
            with open('QM_PR_BACK.potlog', 'w') as f2:
                SubPots.main_inner('PR_NREF.pot', 'QM_PR.pot', out=f, err=f2, subtract=False)

        with open('unfit_diff.log', 'w') as f:
            verbose_call([potential, "5", "QM_PR.xyz", "QM_PR_BACK.pot", "Y"], kwargs={'stdout': f})

        with open('QM_PR.grid', 'r') as f:
            verbose_call([cubegen, '0', 'potential=MP2', 'QM_REF.fchk', 'QM_REF.cube', '-5', 'h'], kwargs={'stdin': f})

        verbose_call([potential, '2', 'QM_REF.cube'])
        SubPots.main_inner('QM_PR.pot', 'QM_REF.pot')
        with open('qm_polarization.pot', 'w') as f:
            with open('polarize_diff_qm.log', 'w') as f2:
                # TODO: Log atom-focal information
                SubPots.main_inner('QM_PR.pot', 'QM_REF.pot', out=f, err=f2, subtract=True, x=foc_atom[2],
                                   y=foc_atom[3], z=foc_atom[4])
        eprint('\n')
        os.chdir("..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='probe_type', type=int, default=999, help='Probe aotm type')
    parser.add_argument('-t', dest='tinker_path', type=str, default='', help='Path to Tinker executables')
    parser.add_argument('-g', dest='gauss_path', type=str, default='', help='Path to Gaussian executables')
    args = parser.parse_args()
    main_inner(tinker_path=args.tinker_path, gauss_path=args.gauss_path)


if __name__ == "__main__":
    main()
