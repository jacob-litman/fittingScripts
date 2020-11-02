#!/usr/bin/env python

import argparse
import re
import os
import sys
import ProbePlacement
import shutil

from ComOptions import ComOptions
from StructureXYZ import StructXYZ
from JMLUtils import eprint
from OptionParser import OptParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='probe_name', type=str, default='PC', help='Atom name to give the probe')
    parser.add_argument('-w', dest='hydrogen_weight', type=float, default=0.4, help='Relative weighting for '
                                                                                    'hydrogen distances')
    parser.add_argument('-e', dest='exp', type=int, default=3, help='Exponent for the square of distance in target '
                                                                    'function (i.e. exp in E = k*separation^(-2*exp)')
    parser.add_argument('-i', dest='infile', type=str, default='ttt.xyz', help='Name of input .xyz file')
    parser.add_argument('--inKey', type=str, default=None, help='Name of input .key file (else from infile)')
    # Required options. Charge/spin are easily forgotten, which is why they're required.
    parser.add_argument('-c', dest='charge', type=int, required=True, help='Charge of the molecule. REQUIRED.')
    parser.add_argument('-s', dest='spin', type=int, required=True, help='Spin of the molecule. REQUIRED.')
    parser.add_argument('-q', dest='probe_charge', type=float, default='0.125', help='Charge to assign to the probe')
    parser.add_argument('-t', dest='tinker_path', type=str, default='', help='Path to Tinker executables')
    parser.add_argument('-o', dest='opts_file', type=str, default=None,
                        help='File containing key=value properties: default locations: poltype.ini, espfit.ini')

    args = parser.parse_args()
    if args.inKey is None:
        keyf = re.sub(r'\.xyz(?:_\d+)?$', '.key', args.infile)
        if not os.path.exists(keyf):
            eprint("No key file found: exiting!")
            sys.exit(1)
    else:
        keyf = args.inKey
    xyz_in = StructXYZ(args.infile, key_file=keyf)
    opts = OptParser(args.opts_file)
    n_physical = xyz_in.n_atoms

    eprint("Step 1: writing reference .com file")
    init_com_opts = ComOptions(args.charge, args.spin, opts=opts)
    jname = 'QM_REF'
    init_com_opts.chk = f"{jname}.chk"
    init_com_opts.do_polar = True
    xyz_in.write_qm_job(com_opts=init_com_opts, fname='QM_REF', jname=jname)
    physical_atom_ids = [f"{xyz_in.atom_names[i]}{i+1}" for i in range(n_physical)]
    # TODO: Customize this via either poltype.ini or similar.

    eprint("Step 2: writing key files with probe (with reference and uncharged solutes)")
    probe_type = xyz_in.append_atype_def(ProbePlacement.DEFAULT_PROBE_TYPE, ProbePlacement.DEFAULT_PROBE_TYPE,
                                         args.probe_name, ProbePlacement.DEFAULT_PROBE_DESC, 1, 1.0, 0, True)

    atype_out = f'atom        {probe_type[0]:>5d}  {probe_type[1]:>5d}  {probe_type[2]:>3s}     {probe_type[3]}      ' \
                f'{probe_type[4]:>4d} {probe_type[5]:>9.3f}   {probe_type[6]:>2d}\n'
    vdw_out = f'vdw     {probe_type[1]:>5d}  0.0100   0.0000\n'
    polarize_out = f'polarize         {probe_type[0]:>5d}          0.0000     0.0100\n'
    pc = args.probe_charge
    multipole_out = f'multipole {probe_type[0]:>5d}                     {pc:>11.5f}\n'
    multipole_out += "                                        0.00000    0.00000    0.00000\n"
    multipole_out += "                                        0.00000\n"
    multipole_out += "                                        0.00000    0.00000\n"
    multipole_out += "                                        0.00000    0.00000    0.00000\n"
    addtl_lines = ['\n\n', '# Below added by EspSetup.py!\n', '', atype_out, vdw_out, multipole_out, polarize_out, '\n']

    xyz_in.write_key_old('QM_PR.key', added_lines=addtl_lines)
    xyz_in.write_key_old_neutral('PR_NREF.key', added_lines=addtl_lines)

    eprint("Step 3: placing probe!")
    probe_locs = ProbePlacement.main_inner(xyz_in, out_file_base='QM_PR', probe_type=probe_type, keyf='QM_PR.key')

    eprint("Step 4: creating probe subdirectories.")
    probe_qm_opt = ComOptions(args.charge, args.spin)
    probe_qm_opt.chk = "QM_PR.chk"
    for i, pid in enumerate(physical_atom_ids):
        dirn = f"{pid}{os.sep}"
        shutil.copy2(f"{dirn}QM_PR.xyz", f"{dirn}PR_NREF.xyz")
        shutil.copy2('PR_NREF.key', dirn)
        shutil.copy2('QM_PR.key', dirn)
        shutil.copy2(keyf, f"{dirn}QM_REF.key")
        xyz_in.coords[n_physical, :] = probe_locs[i, :]
        xyz_in.write_qm_job(probe_qm_opt, f"{dirn}QM_PR.com", "QM_PR", args.probe_charge)


if __name__ == "__main__":
    main()
