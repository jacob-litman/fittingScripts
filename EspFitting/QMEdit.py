import OptionParser
import os
import shutil
import EspSetup
import re
import StructureXYZ
import sys
import JMLUtils

from typing import List, Sequence
from JMLUtils import eprint, extract_molspec
from StructureXYZ import StructXYZ
from ComOptions import ComOptions
from os.path import join

dir_files_psi = frozenset(('PR_NREF.dat', 'QM_PR.key', 'QM_REF.psi4'))


def probable_probe(root: str, dirs: List[str], files: List[str]) -> (bool, str, List[str]):
    root += os.sep
    if 'espfit.ini' in files:
        ini_fi = 'espfit.ini'
    elif 'poltype.ini' in files:
        ini_fi = 'poltype.ini'
    else:
        return False, None, None
    ini_fi = join(root, ini_fi)

    if 'QM_REF.xyz' not in files:
        return False, None, None

    ret_dirs = []

    if 'QM_REF.psi4' in files:
        qm_pr = 'QM_PR.psi4'
    elif 'QM_REF.com' in files:
        qm_pr = 'QM_PR.com'
    else:
        return False, None, None

    for d in dirs:
        d_full = join(root, d)
        if os.path.exists(join(d_full, qm_pr)) and os.path.exists(join(d_full, 'QM_PR.xyz')):
            ret_dirs.append(d_full)
    if len(ret_dirs) == 0:
        return False, None, None
    return True, ini_fi, ret_dirs


def main():
    if len(sys.argv) > 1:
        probe_types = [int(arg) for arg in sys.argv[1:]]
    else:
        probe_types = None
    main_inner(probe_types)


def main_inner(probe_types: Sequence[int] = None):
    if probe_types is None:
        probe_types = [StructureXYZ.DEFAULT_PROBE_TYPE]
    optp = OptionParser.OptParser()
    if optp.is_psi4():
        new_qm_ref = 'QM_REF.psi4'
        new_qm_pr = 'QM_PR.psi4'
    else:
        new_qm_ref = 'QM_REF.com'
        new_qm_pr = 'QM_PR.com'

    orig_program = None
    for root, dirs, files in os.walk('.'):
        if root == ".":
            continue
        is_pdir, ini_fi, subdirs = probable_probe(root, dirs, files)
        if is_pdir:
            with open(ini_fi, 'r') as r:
                for line in r:
                    line = line.replace(" ", "")
                    if line.lower().startswith("program="):
                        orig_program = JMLUtils.parse_qm_program(line.split("=")[1])
    if orig_program is None:
        raise ValueError("Could not find out which program was used previously to set up this directory!")
    elif orig_program == JMLUtils.QMProgram.PSI4:
        old_qm_ref = 'QM_REF.psi4'
        old_qm_pr = 'QM_PR.psi4'
    else:
        old_qm_ref = 'QM_REF.com'
        old_qm_pr = 'QM_PR.com'

    in_file = optp.file
    for root, dirs, files in os.walk('.'):
        if root == ".":
            continue
        is_pdir, ini_fi, subdirs = probable_probe(root, dirs, files)
        if not is_pdir:
            assert ini_fi is None and subdirs is None
            continue
        assert ini_fi is not None and subdirs is not None

        eprint(f"Over-writing {ini_fi} with {in_file}.")
        os.remove(ini_fi)
        shutil.copy2(in_file, ini_fi)

        xyz = StructXYZ(join(root, 'QM_REF.xyz'), probe_types=probe_types)
        old_qm = join(root, old_qm_ref)
        new_qm = join(root, new_qm_ref)
        eprint(f"Replacing {old_qm} with new file {new_qm}")
        mspec = extract_molspec(old_qm, orig_program)
        charge = mspec[0]
        spin = mspec[1]
        eprint(f"Extracted charge/spin: {charge}/{spin}")
        os.remove(old_qm)
        EspSetup.write_init_qm(xyz, charge, spin, optp, fname=new_qm)

        probe_type = probe_types[0]
        probe_comopts = EspSetup.get_probe_comopts(charge, spin, optp)
        for sd in subdirs:
            xyz = StructXYZ(join(sd, 'QM_PR.xyz'), probe_types=probe_types)
            old_qm = join(sd, old_qm_pr)
            new_qm = join(sd, new_qm_pr)
            eprint(f"  Replacing {old_qm} with new file {new_qm}")
            probe_charge = None
            probe_mpole_patt = re.compile(f'^\s*multipole\s+{probe_type}\s.+\s(-?\d+\.\d+)\s*$')
            with open(xyz.key_file, 'r') as r:
                for line in r:
                    m = probe_mpole_patt.match(line)
                    if m:
                        probe_charge = float(m.group(1))
                        break
            assert probe_charge is not None
            eprint(f"  Extracted probe type {probe_type} with charge {probe_charge}")
            os.remove(old_qm)
            EspSetup.write_probe_qm(xyz, optp, probe_comopts, sd, probe_charge)


if __name__ == "__main__":
    main()
