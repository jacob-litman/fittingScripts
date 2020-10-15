#!/usr/bin/env python

import argparse
import re
import os
import sys
import ProbePlacement

from ComOptions import ComOptions
from StructureXYZ import StructXYZ
from JMLUtils import eprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='probe_name', type=str, default='PC', help='Atom name to give the probe')
    parser.add_argument('-w', dest='hydrogen_weight', type=float, default=0.4, help='Relative weighting for '
                                                                                    'hydrogen distances')
    parser.add_argument('-e', dest='exp', type=int, default=3, help='Exponent for the square of distance in target '
                                                                    'function (i.e. exp in E = k*separation^(-2*exp)')
    parser.add_argument('-x', dest='xyzpdb', type=str, default='xyzpdb', help='Name or full path of Tinker '
                                                                              'xyzpdb')
    parser.add_argument('-i', dest='infile', type=str, default='ttt.xyz', help='Name of input .xyz file')
    parser.add_argument('--inKey', type=str, default=None, help='Name of input .key file (else from infile)')
    # Required options. Charge/spin are easily forgotten, which is why they're required.
    parser.add_argument('-c', dest='charge', type=int, required=True, help='Charge of the molecule. REQUIRED.')
    parser.add_argument('-s', dest='spin', type=int, required=True, help='Spin of the molecule. REQUIRED.')
    parser.add_argument('-q', dest='probe_charge', type=float, default='0.125', help='Charge to assign to the probe')

    args = parser.parse_args()
    if args.inKey is None:
        keyf = re.sub(r'\.xyz(?:_\d+)?$', '.key', args.infile)
        if not os.path.exists(keyf):
            eprint("No key file found: exiting!")
            sys.exit(1)
    else:
        keyf = args.inKey
    xyz_in = StructXYZ(args.infile, key_file=keyf)

    eprint("Step 1: writing reference .com file")
    init_com_opts = ComOptions(args.charge, args.spin)
    jname = 'QM_REF'
    init_com_opts.chk = f"{jname}.chk"
    #self, com_opts: ComOptions, fname: str = None, jname: str = None, probe_charge: float = 0.125
    xyz_in.write_com(com_opts=init_com_opts, fname='QM_REF.com', jname=jname)
    # TODO: Customize this via either poltype.ini or similar.


    eprint("Step 2: writing key files with probe (with reference and uncharged solutes)")
    probe_type = xyz_in.append_atype_def(ProbePlacement.DEFAULT_PROBE_TYPE, ProbePlacement.DEFAULT_PROBE_TYPE,
                                         args.probe_name, ProbePlacement.DEFAULT_PROBE_DESC, 1, 1.0, 0, True)

    atype_out = f'atom        {probe_type[0]:>5d}  {probe_type[1]:>5d}  {probe_type[2]:>3s}     {probe_type[3]}      ' \
                f'{probe_type[4]:>4d} {probe_type[5]:>9.3f}   {probe_type[6]:>2d}\n'
    vdw_out = f'vdw     {probe_type[1]:>5d}  0.0100   0.0000\n'
    polarize_out = f'polarize         {probe_type[0]:>5d}          0.0000     0.0100\n'
    '''multipole   405  401  404              -0.08274
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000'''
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
    ProbePlacement.main_inner(xyz_in, out_file_base='QM_PR', probe_type=probe_type)



if __name__ == "__main__":
    main()
