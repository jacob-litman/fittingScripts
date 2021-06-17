import PolarTypeReader
import StructureXYZ
import os
import re
from os.path import join
from EspFitPolar import molec_dir_patt, check_finished_qm
from JMLUtils import eprint

def main_inner():
    ptyping = PolarTypeReader.PtypeReader('polarTypes.tsv')
    ref_files = []
    with open('molecules.txt', 'r') as r:
        for line in r:
            line = re.sub(r'#.+', '', line.strip()).strip()
            if line != "" and os.path.isfile(line):
                ref_files.append(line)

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

    ptype_mols = dict()
    for pt in ptyping.ptypes:
        ptype_mols[pt] = set()

    for dir in ref_dirs:
        qmr = join(dir, 'QM_REF.xyz')
        sqmr = StructureXYZ.StructXYZ(qmr)

        ptypes_i, mapping_i, pt2 = PolarTypeReader.main_inner(sqmr, False, ptyping=ptyping)
        for pt_i in ptypes_i:
            ptype_mols[pt_i].add(dir)

    print(f"Polar type\tSMARTS\tmolecules...")
    for k, v in ptype_mols.items():
        if len(v) > 0:
            print(f"{k.name}\t{k.format_smarts()}", end='')
            for mol in v:
                print(f"\t{mol}", end='')
            print('')


def main():
    main_inner()


if __name__ == "__main__":
    main()