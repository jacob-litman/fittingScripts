import subprocess
import sys

import numpy as np
import re
from typing import Sequence
from pathlib import Path
from io import TextIOWrapper

from ComOptions import ComOptions
from JMLUtils import eprint, version_file, cryst_patt, verbose_call
import os

# Captured: atom type, atom class, atom name, atom description, atomic number, mass, valency
adef_patt = re.compile(r'^atom +(\d+) +(\d+) +(\S+) +"?([^"]+)"? +(\d+) +(\d+\.\d+) +(\d+)\s*$')
mpole_cont_patt = re.compile(r'^( +-?\d+\.\d+)+ *\\? *$')
polarize_patt = re.compile(r'^(polarize +\d+ +)(\d+\.\d+ +\d+\.\d+)( [ 0-9]+ *)$')
DEFAULT_PROBE_ANUM = 999
DEFAULT_PSI4_ITERS = 300
# TODO: Either use the periodictable package (Pip) or flesh this out.
elements = {1: 'H', 2: 'HE',
            3: 'LI', 4: 'BE', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'NE',
            11: 'NA', 12: 'MG', 13: 'AL', 14: 'SI', 15: 'P', 16: 'S', 17: 'CL', 18: 'AR'}

DEFAULT_PROBE_DESC = "Probe Charge        "
DEFAULT_PROBE_NAME = 'PC'
DEFAULT_PROBE_TYPE = 999
DEFAULT_PROBE_MASS = 1.0

class StructXYZ:
    def __init__(self, in_file: str, probe_types: Sequence[int] = None, key_file: str = None):
        self.in_file = in_file
        with open(self.in_file, 'r') as f:
            in_line = f.readline()
            self.header_line = in_line
            self.n_atoms = int(in_line.strip().split()[0])
            skip_r = 1
            in_line = f.readline()
            cryst_match = cryst_patt.match(in_line)
            if cryst_match:
                skip_r = 2
                self.cryst_info = [float(cryst_match.group(i)) for i in range(1, 7, 1)]
                in_line = f.readline()
            else:
                self.cryst_info = None
            self.aperiodic = self.cryst_info is None

            self.bonds_bidi = []
            while in_line != "":
                toks = in_line.strip().split()
                bonds_this = []
                for i in range(6, len(toks), 1):
                    bonds_this.append(int(toks[i]))
                self.bonds_bidi.append(bonds_this)
                in_line = f.readline()

        self.atom_numbers = np.genfromtxt(self.in_file, skip_header=skip_r, usecols=[0], dtype=np.int32)
        self.atom_names = np.genfromtxt(self.in_file, skip_header=skip_r, usecols=[1], dtype=str)
        self.coords = np.genfromtxt(self.in_file, skip_header=skip_r, usecols=(2, 3, 4), dtype=np.float64)
        self.assigned_atypes = np.genfromtxt(self.in_file, skip_header=skip_r, usecols=[5], dtype=np.int32)
        self.default_probetype = None

        self.probe_indices = []
        if probe_types is None:
            self.probe_types = []
        else:
            self.probe_types = list(probe_types)
            for i in range(self.n_atoms):
                if self.assigned_atypes[i] in self.probe_types:
                    self.probe_indices.append(i)

        if key_file is None:
            test_key = re.sub(r'\.xyz(?:_\d+)?$', '.key', self.in_file)
            if os.path.exists(test_key):
                key_file = test_key

        self.key_file = key_file
        if key_file is None:
            eprint("Failed to find a corresponding key file: some information (e.g. elements) not read in!")
            self.def_atypes = None
        else:
            self.def_atypes = dict()
            with open(key_file, 'r') as f:
                in_line = f.readline()
                while in_line != '':
                    if in_line.startswith("atom  "):
                        m = adef_patt.match(in_line)
                        atype = int(m.group(1))
                        # Captured: atom type, atom class, atom name, atom description, atomic number, mass, valency
                        new_atype = (atype, int(m.group(2)), m.group(3), m.group(4), int(m.group(5)),
                                     float(m.group(6)), int(m.group(7)))
                        self.def_atypes[atype] = new_atype
                        # TODO: Read more information.
                    in_line = f.readline()

    def __str__(self):
        if self.aperiodic:
            ret_str = f'Aperiodic system {self.in_file}'
        else:
            ret_str = f'Periodic system {self.in_file}'
        if self.key_file is None:
            return f'{ret_str} with no known key file'
        else:
            return f'{ret_str} with key file {self.key_file}'

    def get_atype_info(self, i: int) -> (int, int, str, str, int, float, int):
        assert self.def_atypes is not None
        return self.def_atypes[self.assigned_atypes[i]]

    def append_atom(self, atype: int, xyz: np.ndarray, bonds: Sequence[int] = None):
        if isinstance(atype, tuple):
            # Assume a whole probetype tuple was passed, not just the index.
            atype = atype[0]
        name = self.def_atypes[atype][2]
        if atype in self.probe_types:
            self.probe_indices.append(self.n_atoms)
        self.n_atoms += 1
        self.atom_numbers = np.append(self.atom_numbers, [self.n_atoms])
        self.atom_names = np.append(self.atom_names, [name])
        self.coords = np.append(self.coords, [xyz], axis=0)
        self.assigned_atypes = np.append(self.assigned_atypes, [atype])
        if bonds is None:
            bonds = []
        self.bonds_bidi.append(bonds)

    def append_atype_def(self, atype: int, aclass: int, aname: str, desc: str, anum: int, mass: float, valency: int,
                         isprobe: bool = False) -> (int, int, str, str, int, float, int):
        assert anum > 0 and mass > 0 and valency >= 0
        max_aclass = -1
        max_atype = -1
        aclass_found = False
        atype_found = False

        for atype2 in self.def_atypes.values():
            at2 = atype2[0]
            ac2 = atype2[1]
            max_atype = max(max_atype, at2)
            max_aclass = max(max_aclass, ac2)
            atype_found = atype_found or at2 == atype
            aclass_found = aclass_found or ac2 == aclass

        if atype < 0 or atype_found:
            eprint(f"Atom type {atype} already defined; defining atom {aname}-{desc} as type {max_atype + 1}")
            atype = max_atype + 1
        if aclass < 0 or aclass_found:
            eprint(f"Atom class {aclass} already defined; defining atom {aname}-{desc} as class {max_aclass + 1}")
            aclass = max_aclass + 1

        new_atype_def = (atype, aclass, aname, desc, anum, mass, valency)
        self.def_atypes[atype] = new_atype_def
        if isprobe:
            assert atype not in self.probe_types
            self.probe_types.append(atype)
            for i in range(self.n_atoms):
                if self.assigned_atypes[i] == atype:
                    assert i not in self.probe_indices
                    self.probe_indices.append(i)
        return new_atype_def

    def write_out(self, fname: str = None):
        if fname is None:
            fname = version_file(self.in_file)
        with open(fname, 'w') as f:
            f.write(f"{self.n_atoms:7d}  {fname}")
            header_toks = self.header_line.strip().split()
            n_ht = len(header_toks)
            for i in range(2, n_ht, 1):
                f.write(f" {header_toks[i]:s}")
            f.write('\n')
            if not self.aperiodic:
                for ci in self.cryst_info:
                    f.write(f"{ci:14.8f}")
                f.write('\n')
            for i in range(self.n_atoms):
                xi = self.coords[i]
                # Slight difference from FFX reference: instead of "14.8f", use " 13.8f" with a space
                # This ensures there's always a space between fields, even if there's overflow.
                f.write(f"{self.atom_numbers[i]:7d} {self.atom_names[i]:3s} {xi[0]:13.8f} {xi[1]:13.8f} {xi[2]:13.8f} "
                        f"{self.assigned_atypes[i]:5d}")
                for b in self.bonds_bidi[i]:
                    f.write(f" {b:7d}")
                f.write('\n')

    def format_atom(self, atom_index: int) -> str:
        ret_str = f"Atom {self.atom_names[atom_index]}-{atom_index + 1:d}, atom type {self.assigned_atypes[atom_index]}"
        ret_str += f" at position"
        for i in range(3):
            ret_str += f" {self.coords[atom_index][i]:.4f}"
        return ret_str

    def write_key_old(self, fname: str, added_lines: Sequence[str] = None):
        with open(self.key_file, 'r') as k:
            with open(fname, 'w') as f:
                f.writelines(k.readlines())
                f.writelines(added_lines)

    # TODO: Do this in less terrible fashion and merge w/ write_key
    def write_key_old_neutral(self, fname: str, added_lines: Sequence[str] = None):
        '''Writes a key file with optional additional lines while neutralizing non-probe electrostatics'''
        with open(self.key_file, 'r') as k:
            with open(fname, 'w') as f:
                line = k.readline()
                while line != '':
                    if line.startswith('multipole'):
                        toks = line.split()
                        atype = int(toks[1])
                        if atype in self.probe_types:
                            f.write(line)
                            for i in range(4):
                                f.write(k.readline())
                        else:
                            # Discard the next four lines.
                            for i in range(4):
                                k.readline()
                            n_tok = len(toks)
                            f.write("multipole ")
                            for i in range(1, n_tok - 1, 1):
                                f.write(f"{toks[i]:>5s}")
                            f.write("               0.00000\n")
                            f.write("                                        0.00000    0.00000    0.00000\n")
                            f.write("                                        0.00000\n")
                            f.write("                                        0.00000    0.00000\n")
                            f.write("                                        0.00000    0.00000    0.00000\n")

                    elif line.startswith('polarize'):
                        # Probe should have no polarizability anyways, so neuter everything.
                        m = polarize_patt.match(line)
                        f.write(m.group(1))
                        f.write("0.0000     0.0100")
                        f.write(m.group(3) + "\n")
                        pass

                    else:
                        f.write(line)

                    line = k.readline()
                f.writelines(added_lines)

    def write_qm_job(self, com_opts: ComOptions, fname: str = None, jname: str = None, probe_charge: float = 0.125,
                     header_lines: Sequence[str] = None, footer_lines: Sequence[str] = None,
                     write_fchk: bool = False) -> str:
        """Writes a QM job (either .com for Gaussian or .psi4 for Psi4) based on this structure"""
        if fname is None:
            fname = f"{os.path.splitext(self.in_file)[0]}_QM"
            if jname is None:
                jname = fname
        elif jname is None:
            jname = os.path.splitext(fname)[0]

        if com_opts.program == 'PSI4':
            if not fname.endswith('.psi4'):
                fname += '.psi4'
            self.write_psi4_input(com_opts, fname=fname, jname=jname, probe_charge=probe_charge, header_lines=header_lines, footer_lines=footer_lines, write_fchk=write_fchk)
        elif com_opts.program.startswith('GAUSS'):
            if not fname.endswith('.com'):
                fname += '.com'
            self.write_com(com_opts, fname=fname, jname=jname, probe_charge=probe_charge, header_lines=header_lines, footer_lines=footer_lines, write_fchk=write_fchk)
        return fname

    def write_coords(self, f: TextIOWrapper, com_opts: ComOptions):
        f.write(f"{com_opts.charge} {com_opts.spin}\n")
        for i in range(self.n_atoms):
            if i in self.probe_indices:
                # eprint(f"Atom is likely a probe: not written as an atom: {self.format_atom(i)}")
                continue
            el = self.get_atype_info(i)[4]
            el_name = elements[el]
            f.write(f"{el_name:>2s}")
            for j in range(3):
                f.write(f"{self.coords[i][j]:12.6f}")
            f.write('\n')

    def write_psi4_input(self, com_opts: ComOptions, fname: str, jname: str, probe_charge: float = 0.125,
                         header_lines: Sequence[str] = None, footer_lines: Sequence[str] = None,
                         write_fchk: bool = False):
        assert com_opts.rwf is not None
        if jname is None:
            jname = f"{Path(fname).stem}_QM"
        with open(fname, 'w') as f:
            if header_lines is not None:
                for line in header_lines:
                    f.write(line)
                    f.write('\n')
                f.write('\n')

            f.write("molecule {\n")
            self.write_coords(f, com_opts=com_opts)
            f.write("}\n\n")
            f.write(f"memory {com_opts.mem}\n")
            f.write(f"set_num_threads({com_opts.nproc})\n")
            f.write(f'psi4_io.set_default_path("{com_opts.rwf}")\n')
            f.write(f"set maxiter {DEFAULT_PSI4_ITERS}\n")
            # Note: Gaussian does frozen-core by default.
            f.write("set freeze_core True\n")
            # Probably also Gaussian-default.
            f.write('set PROPERTIES_ORIGIN ["COM"]\n\n')
            if self.probe_indices is not None and len(self.probe_indices) > 0:
                f.write('Chrgfield = QMMM()\n')
                for pi in self.probe_indices:
                    f.write(f'Chrgfield.extern.addCharge({probe_charge:f}')
                    for i in range(3):
                        f.write(f',{self.coords[pi][i]:.6f}')
                    f.write(')\n')
                f.write("psi4.set_global_option_python('EXTERN', Chrgfield.extern)\n")

            property_list = []
            if com_opts.write_esp:
                property_list.append('GRID_ESP')
            if com_opts.do_polar:
                eprint("Psi4 cannot calculate molecular polarizabilities for currently used methods (MP2, DFT); obtain "
                       "these values separately!")

            read_chk = False
            if com_opts.chk is not None and os.path.exists(com_opts.chk):
                read_chk = True
                f.write(f"wfn = psi4.core.Wavefunction.from_file({com_opts.chk})\n")
                if len(property_list) > 0:
                    f.write(f'oeprop(wfn')
                    for p in property_list:
                        f.write(f', {p}')
                    f.write(')\n')
                else:
                    eprint(f"WARNING: Wavefunction file {com_opts.chk} found, but no properties are specified!")
                    f.write('wfn.compute_energy()\n')
            elif len(property_list) > 0:
                f.write(f"E, wfn = properties('{com_opts.method}/{com_opts.basis}', ")
                f.write("properties=[")
                pstr = ""
                for p in property_list:
                    pstr += f"'{p}', "
                f.write(re.sub(r', $', '], ', pstr))
                f.write('return_wfn=True)\n')
            else:
                f.write(f"E, wfn = energy('{com_opts.method}/{com_opts.basis}, return_wfn=True)\n")

            if write_fchk:
                f.write(f'fchk(wfn, "{jname}.fchk")\n')

            if com_opts.chk is not None:
                if read_chk:
                    eprint(f"NOTE: Over-writing {com_opts.chk} with newly computed wavefunction (if any).")
                f.write(f"wfn.to_file('{com_opts.chk}')\n")

            if footer_lines is not None:
                f.write('\n')
                for line in footer_lines:
                    f.write(line)
                    f.write('\n')

            f.write('\nclean()\n')

    def write_com(self, com_opts: ComOptions, fname: str, jname: str, probe_charge: float = 0.125,
                  header_lines: Sequence[str] = None, footer_lines: Sequence[str] = None, write_fchk: bool = False):
        assert com_opts.rwf is not None

        if header_lines is not None or footer_lines is not None:
            eprint("WARNING: Writing of Gaussian files (.com) currently doesn't support additional header/footer lines!")
        if write_fchk:
            eprint("WARNING: Use the formchk executable separately to write out .fchk files!")

        if jname is None:
            jname = f"{Path(fname).stem}_QM"
        with open(fname, 'w') as f:
            f.write(f"%rwf={com_opts.rwf},{com_opts.storage}\n")
            f.write(f"%mem={com_opts.mem}\n")
            f.write(f"%nproc={com_opts.nproc}\n")
            if com_opts.chk is not None:
                f.write(f"%Chk={com_opts.chk}\n")
            last_header = f"#{com_opts.method}/{com_opts.basis} "
            if com_opts.do_polar:
                last_header += "Polar "
            if com_opts.scf is not None:
                last_header += f"SCF={com_opts.scf} "
            if com_opts.density is not None:
                last_header += f"Density={com_opts.density} "
            if com_opts.guess is not None:
                last_header += f"Guess={com_opts.guess} "
            if com_opts.no_symm:
                last_header += "NoSymm "
            if len(self.probe_indices) > 0:
                last_header += "Charge "
            last_header = last_header.rstrip()
            f.write(f"{last_header}\n\n")
            f.write(f"{jname}\n\n")
            self.write_coords(f, com_opts=com_opts)

            f.write('\n')
            if self.probe_indices is not None and len(self.probe_indices) > 0:
                for pi in self.probe_indices:
                    for i in range(3):
                        f.write(f"{self.coords[pi][i]:.6f} ")
                    f.write(f"{probe_charge:f}\n")
                f.write('\n')

    def get_esp_file(self, potential: str = 'potential', keyf: str = None, verbose: bool = False) -> str:
        if keyf is None:
            keyf = self.key_file
        if verbose:
            eprint(f"Calling (with input capture): potential 3 {self.in_file} {keyf} Y {keyf}")
        sp_args = [potential, '3', self.in_file, "-k", keyf, 'Y']
        output = subprocess.check_output(sp_args)
        for line in output.splitlines():
            line = str(line, encoding=sys.getdefaultencoding()).strip()
            if line.startswith('Electrostatic Potential Written To'):
                esp_fi = re.sub('^Electrostatic Potential Written To : +', '', line)
                return esp_fi
        raise RuntimeError(f'Was unable to find the file electrostatic potential was written to for system {self}')

    def get_esp(self, potential: str = 'potential', keyf: str = None, delete_file: bool = True) -> np.ndarray:
        esp_fi = self.get_esp_file(potential=potential, keyf=keyf)
        esp_arr = np.genfromtxt(esp_fi, skip_header=1, usecols=(1, 2, 3, 4))
        if delete_file:
            os.remove(esp_fi)
        return esp_arr

    def get_default_probetype(self):
        if self.default_probetype is None:
            self.default_probetype = self.append_atype_def(DEFAULT_PROBE_TYPE, DEFAULT_PROBE_TYPE, DEFAULT_PROBE_NAME,
                                                           DEFAULT_PROBE_DESC, DEFAULT_PROBE_ANUM, DEFAULT_PROBE_MASS,
                                                           0, isprobe=True)
        return self.default_probetype
