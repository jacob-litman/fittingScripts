import OptionParser
import os
import shutil
import EspSetup

from typing import List
from JMLUtils import eprint, extract_molspec
from StructureXYZ import StructXYZ
from ComOptions import ComOptions

dir_files_psi = frozenset(('PR_NREF.dat', 'QM_PR.key', 'QM_REF.psi4'))


def probable_probe(root: str, dirs: List[str], files: List[str], optp: OptionParser.OptParser) -> (bool, str, List[str]):
    root += os.sep
    is_psi4 = optp.is_psi4()
    if 'espfit.ini' in files:
        ini_fi = 'espfit.ini'
    elif 'poltype.ini' in files:
        ini_fi = 'poltype.ini'
    else:
        return False, None, None
    ini_fi = os.path.join(root, ini_fi)

    ret_dirs = []
    if is_psi4:
        qm_ref = 'QM_REF.psi4'
        qm_pr = 'QM_PR.psi4'
    else:
        # Assumed Gaussian
        qm_ref = 'QM_REF.com'
        qm_pr = 'QM_PR.com'

    if qm_ref not in files or 'QM_REF.xyz' not in files:
        return False, None
    for d in dirs:
        d_full = os.path.join(root, d)
        if os.path.exists(os.path.join(d_full, qm_pr)) and os.path.exists(os.path.join(d_full, 'QM_PR.xyz')):
            ret_dirs.append(d_full)
    if len(ret_dirs) == 0:
        return False, None, None
    return True, ini_fi, ret_dirs


def main():
    optp = OptionParser.OptParser()
    if optp.is_psi4():
        qm_ref = 'QM_REF.psi4'
        qm_pr = 'QM_PR.psi4'
    else:
        qm_ref = 'QM_REF.com'
        qm_pr = 'QM_PR.com'

    in_file = optp.file
    for root, dirs, files in os.walk('.'):
        if root == ".":
            continue
        is_pdir, ini_fi, subdirs = probable_probe(root, dirs, files, optp)
        if not is_pdir:
            assert ini_fi is None and subdirs is None
            continue
        assert ini_fi is not None and subdirs is not None
        eprint(f"Over-writing {ini_fi} with {in_file}.")
        os.remove(ini_fi)
        shutil.copy2(in_file, ini_fi)
        xyz = StructXYZ(os.path.join(root, 'QM_REF.xyz'))
        qm_fi = os.path.join(root, qm_ref)
        mspec = extract_molspec(qm_fi, optp.get_program())
        charge = mspec[0]
        spin = mspec[1]
        eprint(f"Over-writing {qm_fi} with new options.")
        os.remove(qm_fi)
        EspSetup.write_init_qm(xyz, charge, spin, optp)


if __name__ == "__main__":
    main()
