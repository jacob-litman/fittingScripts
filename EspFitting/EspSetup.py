#!/usr/bin/env python

import argparse
import re
import os
import sys
import ProbePlacement
import shutil

from ComOptions import ComOptions
from StructureXYZ import StructXYZ
from JMLUtils import eprint, verbose_call
from OptionParser import OptParser
from typing import List


def write_init_qm(xyz_in: StructXYZ, charge: int, spin: int, opts: OptParser):
    copts = ComOptions(charge, spin, opts)
    jname = 'QM_REF'
    is_psi4 = opts.is_psi4()
    if is_psi4:
        copts.chk = jname + ".npy"
    else:
        copts.chk = jname + ".chk"
    copts.do_polar = True
    copts.write_esp = True
    xyz_in.write_qm_job(com_opts=copts, fname=jname, jname=jname, write_fchk=True)


def get_probe_comopts(charge: int, spin: int, opts: OptParser) -> ComOptions:
    probe_qm_opt = ComOptions(charge, spin, opts)
    probe_qm_opt.write_esp = True
    if opts.is_psi4():
        probe_qm_opt.chk = 'QM_PR.npy'
    else:
        probe_qm_opt.chk = 'QM_PR.chk'
    return probe_qm_opt


def write_probe_qm(xyz_in: StructXYZ, opts: OptParser, comopts: ComOptions, dirn: str, probe_charge: float, extra_headers: List[str] = None, extra_footers: List[str] = None):
    if extra_footers is None:
        extra_footers = []
    if extra_headers is None:
        extra_headers = []
    is_psi4 = opts.is_psi4()
    if is_psi4:
        extra_headers.append('import shutil')
        extra_footers.extend(['wfn2 = psi4.core.Wavefunction.from_file("../QM_REF.npy")',
                              'shutil.move("grid_esp.dat", "QM_PR.grid_esp.dat")',
                              'oeprop(wfn2, "GRID_ESP")',
                              'shutil.move("grid_esp.dat", "QM_REF.grid_esp.dat")'])

    xyz_in.write_qm_job(comopts, f"{dirn}QM_PR", 'QM_PR', probe_charge, header_lines=extra_headers,
                        footer_lines=extra_footers, write_fchk=True)


def main():
    parser = argparse.ArgumentParser()
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

    if opts.options["program"].upper().startswith("GAUSS"):
        grid_file = 'QM_REF.grid'
        is_psi4 = False
    elif opts.options["program"].upper().startswith("PSI4"):
        grid_file = 'grid.dat'
        is_psi4 = True
    else:
        raise ValueError("Could not determine if program in use is Gaussian or Psi4!")

    eprint("Step 1: writing reference QM input file")
    write_init_qm(xyz_in, args.charge, args.spin, opts)
    physical_atom_ids = [f"{xyz_in.atom_names[i]}{i+1}" for i in range(n_physical)]
    # TODO: Customize this via either poltype.ini or similar.

    eprint("Step 2: writing key files with probe (with reference and uncharged solutes)")
    probe_type = xyz_in.get_default_probetype()

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

    eprint("Step 3: placing probe")
    probe_locs = ProbePlacement.main_inner(xyz_in, out_file_base='QM_PR', probe_type=probe_type[0], keyf='QM_PR.key')

    eprint("Step 4: creating probe subdirectories")
    probe_qm_opt = get_probe_comopts(args.charge, args.spin, opts)

    for i, pid in enumerate(physical_atom_ids):
        dirn = f"{pid}{os.sep}"
        shutil.copy2(f"{dirn}QM_PR.xyz", f"{dirn}PR_NREF.xyz")
        shutil.copy2('PR_NREF.key', dirn)
        shutil.copy2('QM_PR.key', dirn)
        shutil.copy2(keyf, f"{dirn}QM_REF.key")
        xyz_in.coords[n_physical, :] = probe_locs[i, :]
        write_probe_qm(xyz_in, opts, probe_qm_opt, dirn, args.probe_charge)

    tinker_path = args.tinker_path
    if not tinker_path.endswith(os.sep) and tinker_path != '':
        tinker_path += os.sep
    potential = tinker_path + 'potential'
    eprint(f"Step 5: linking {args.infile} to QM_REF.xyz and writing out {grid_file}")
    os.symlink(args.infile, 'QM_REF.xyz')
    os.symlink(keyf, 'QM_REF.key')
    verbose_call([potential, '1', 'QM_REF.xyz', 'QM_REF.key'])
    if is_psi4:
        os.symlink('QM_REF.grid', grid_file)

    if not is_psi4:
        grid_file = 'QM_PR.grid'
    eprint(f"Step 6: Writing out {grid_file} in probe subdirectories")
    for pid in physical_atom_ids:
        dirn = f"{pid}{os.sep}"
        os.chdir(dirn)
        verbose_call([potential, '1', 'QM_PR.xyz', 'QM_PR.key'])
        if is_psi4:
            os.symlink('QM_PR.grid', grid_file)
        os.chdir("..")


if __name__ == "__main__":
    main()
