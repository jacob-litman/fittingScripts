#!/usr/bin/env python

import argparse
from JMLUtils import eprint, get_probe_dirs
from typing import Sequence
import os


def main_inner(probe_name: str, in_name: str = None, out_name: str = "all_probes.pdb", probe_dirs: Sequence[str] = None):
    if probe_dirs is None:
        probe_dirs = get_probe_dirs()
    if in_name is None:
        pdb_files = []
        for dir in probe_dirs:
            pdb_files.extend([fi.name for fi in os.scandir(dir) if (fi.is_file() and fi.name.ends_with(".pdb"))])
    else:
        pdb_files = list(filter(lambda fi: os.path.exists(fi) and os.path.isfile(fi), [f'{pd}{os.sep}{in_name}'
                                                                                       for pd in probe_dirs]))
    assert len(pdb_files) > 0

    base_atom_lines = []
    base_conect_lines = []
    max_atom_num = -1
    max_res_num = -1
    chains = set()
    with open(pdb_files[0], 'r') as r:
        for line in r:
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                at_name = line[12:16].strip()
                if at_name != probe_name:
                    at_serial = int(line[6:11].strip())
                    res_serial = int(line[22:26].strip())
                    max_atom_num = max(max_atom_num, at_serial)
                    max_res_num = max(max_res_num, res_serial)
                    chains.add(line[21])
                    base_atom_lines.append(line)
                else:
                    assert line.startswith("HETATM")
            elif line.startswith("CONECT"):
                base_conect_lines.append(line)

    n_chain = len(chains)
    assert n_chain > 0
    chainlist = list(chains)
    if chainlist[0] == ' ':
        if n_chain > 1:
            out_chain = chainlist[1]
        else:
            eprint(f'No non-blank chains detected; giving probe atoms chain P.')
            out_chain = 'P'
    else:
        out_chain = chainlist[0]

    with open(out_name, 'w') as w:
        for line in base_atom_lines:
            w.write(line)
        for pdbf in pdb_files:
            with open(pdbf, 'r') as r:
                at_ctr = 0
                conect_ctr = 0
                for line in r:
                    if line.startswith("HETATM") or line.startswith("ATOM  "):
                        at_name = line[12:16].strip()
                        if at_name == probe_name:
                            assert line.startswith("HETATM")
                            max_atom_num += 1
                            max_res_num += 1
                            w.write(f'HETATM{max_atom_num:>5d}{line[11:21]}{out_chain}{max_res_num:>4d}'
                                    f'{line[26:]}')
                        else:
                            if not line == base_atom_lines[at_ctr]:
                                eprint(f"WARNING: Non-probe atom {at_ctr + 1} of file {pdbf} does not match atom "
                                       f"{at_ctr + 1} of baseline file {pdb_files[0]}")
                                eprint(f"Problem Line:   {line}")
                                eprint(f"Reference Line: {base_atom_lines[at_ctr]}")
                            at_ctr += 1
                    elif line.startswith("CONECT"):
                        if not line == base_conect_lines[conect_ctr]:
                            eprint(f"WARNING: Non-probe CONECT record {conect_ctr + 1} of file {pdbf} does not match "
                                   f"CONECT {conect_ctr + 1} of baseline file {pdb_files[0]}")
                            eprint(f"Problem Line:   {line}")
                            eprint(f"Reference Line: {base_atom_lines[conect_ctr]}")
                        conect_ctr += 1

        for line in base_conect_lines:
            w.write(line)
        w.write('END\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='probe_name', type=str, default='PC', help='Name of probe atoms.')
    parser.add_argument('-f', dest='file_name', type=str, default=None, help='Base name (excluding directory) of PDB '
                                                                             'files to read.')
    args = parser.parse_args()
    main_inner(args.probe_name, in_name=args.file_name)

if __name__ == "__main__":
    main()
