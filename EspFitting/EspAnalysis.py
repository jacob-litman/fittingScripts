#!/usr/bin/env python

import argparse
import os
import subprocess
from JMLUtils import eprint, verbose_call

#def check_probedir(d: os.DirEntry) -> bool:
#    if d.is_dir():


def main_inner(tinker_path: str = '', gauss_path: str = ''):
    probe_dirs = [f.path for f in os.scandir(".") if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]

    if not tinker_path.endswith(os.sep):
        tinker_path += os.sep
    potential = tinker_path + "potential"

    if not gauss_path.endswith(os.sep):
        gauss_path += os.sep
    formchk = gauss_path + "formchk"
    cubegen = gauss_path + "cubegen"

    for pdir in probe_dirs:
        os.chdir(pdir)
        verbose_call([potential, "1", "QM_PR.xyz"])
        verbose_call([formchk, "QM_PR.chk"])
        # TODO: Check that it actually is MP2 potential.
        with open('QM_PR.grid', 'r') as f:
            #eprint(f"Calling {cubegen} 0 potential=MP2 QM_PR.fchk QM_PR.cube -5 h < MM_PR.grid")
            #subprocess.run([cubegen, '0', 'potential=MP2', 'QM_PR.fchk', 'QM_PR.cube', '-5', 'h'], stdin=f)
            verbose_call([cubegen, '0', 'potential=MP2', 'QM_PR.fchk', 'QM_PR.cube', '-5', 'h'], kwargs={'stdin': f})
        verbose_call([potential, "2", "QM_PR.cube"])
        verbose_call([potential, "3", "PR_NREF.xyz", "Y"])
        # TODO: Bring in AddPots.py: AddPots.py "${PR_NREF}.pot" "${QM_PR}.pot" > "${QM_PR_BACK}.pot"
        with open('unfit_diff.log', 'w') as f:
            verbose_call([potential, "5", "MM_PR.xyz", "QM_PR_BACK.pot", "Y"], kwargs={'stdout': f})
        with open('QM_PR.grid', 'r') as f:
            verbose_call([cubegen, '0', 'potential=MP2', 'QM_REF.fchk', 'QM_REF.cube', '-5', 'h'], kwargs={'stdin': f})
        verbose_call([potential, '2', 'QM_REF.cube'])
        # TODO: Bring in SubPots.py: SubPots.py "${QM_PR}.pot" "${QM_REF}.pot" X Y Z Radius 1> qm_polarization.pot 2> polarize_diff_qm.log
        
        os.chdir("..")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='probe_type', type=int, default=999, help='Probe aotm type')
    parser.add_argument('-t', dest='tinker_path', type=str, default=None, help='Path to Tinker executables')
    parser.add_argument('-g', dest='gauss_path', type=str, default=None, help='Path to Gaussian executables')
    args = parser.parse_args()
    main_inner(tinker_path=args.tinker_path)

if __name__ == "__main__":
    main()