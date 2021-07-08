import sys
import os
import subprocess
import re
import math
from typing import Sequence, Dict, Mapping, List
from enum import Enum, auto
import numpy as np

cryst_patt = re.compile(r"^ +(-?\d+\.\d+){6} *$")
chargespin_patt = re.compile(r"^ *(-?\d+) +(-?\d+) *$")
hartree = 627.5094736
gauss_polar_convert = 0.52917721067 ** 3


def eprint(*args, kwargs: dict = None):
    """Prints to stderr; effectively wraps the print() function with kwargs['file'] = sys.stderr, and kwargs['flush'] =
    True unless already defined in kwargs."""
    if kwargs is None:
        kwargs = dict()
    kwargs['file'] = sys.stderr
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    print(*args, **kwargs)


def verbose_call(args: Sequence[str], kwargs = None, to_stderr: bool = True):
    out_str = "Calling:"
    for a in args:
        out_str += f" {a}"
    if to_stderr:
        eprint(out_str)
    else:
        print(out_str)
    if kwargs is None:
        kwargs = dict()
    subprocess.run(args, **kwargs)


def version_file(fname: str) -> str:
    if os.path.exists(fname):
        for i in range(1000):
            trial_fname = f"{fname}_{i}"
            if not os.path.exists(trial_fname):
                return trial_fname
        raise RuntimeError(f"Could not version {fname} to {fname}_x for 1<=x<=1000!")
    else:
        return fname


def name_to_atom(xyzf: str, name: str) -> (int, str, float, float, float, int, Sequence[int]):
    with open(xyzf, 'r') as f:
        f.readline()
        line = f.readline()
        if (cryst_patt.match(line)):
            line = f.readline()
        while line != '':
            toks = line.split()
            test_str = toks[1] + toks[0]
            if test_str == name:
                bonds = [int(t) for t in toks[6:]]
                return int(toks[0]), toks[1], float(toks[2]), float(toks[3]), float(toks[4]), int(toks[5]), bonds
            line = f.readline()
    eprint(f"No match for {name} found in {xyzf}!")
    return None


def audit_files(files: Mapping[str, float] = None) -> (Mapping[str, float], Mapping[str, float]):
    # Files may be None if attempting to generate the first mapping.
    if files is None:
        files = dict()

    modified = dict()
    added = dict()
    for fi in os.scandir():
        if fi.is_file():
            fi_name = fi.name
            if fi_name in files:
                mtime = fi.stat().st_mtime
                if mtime != files[fi_name]:
                    modified[fi_name] = mtime
            else:
                added[fi_name] = fi.stat().st_mtime
    return modified, added


def log_audit_files(files: Dict[str, float], about: str):
    modified, added = audit_files(files)
    log_audit_files.n_log_audit += 1
    eprint(f"AUDIT {log_audit_files.n_log_audit:>3d}: ", kwargs={'end': ''})

    if len(modified.keys()) == 0 and len(added.keys()) == 0:
        eprint(f"No effect from {about}")
        return

    if len(modified) > 0:
        eprint(f"Modified since {about}:")
        for m, t in modified.items():
            eprint(f"Updated timestamp on {m}: {files[m]:.6f} -> {t:.6f}")
    if len(added) > 0:
        eprint(f"Added since {about}:")
        for a, t in added.items():
            eprint(f"Added file {a} with timestamp {t:.6f}")

    # Now, update the mapping.
    for m, t in modified.items():
        files[m] = t
    for a, t in added.items():
        files[a] = t
    eprint("\n")
log_audit_files.n_log_audit = 0


def dist2(coord1: np.ndarray, coord2: np.ndarray) -> float:
    return np.square(coord1 - coord2).sum()


def get_probe_dirs(base_dir: str = ".") -> List[str]:
    return [f.path for f in os.scandir(base_dir) if (f.is_dir() and os.path.exists(f"{f.path}{os.sep}QM_PR.xyz"))]


class QMProgram(Enum):
    GAUSSIAN = tuple("GAUSS_SCRDIR")
    PSI4 = tuple("PSI4_SCRATCH")

    def get_scrdir_name(self) -> str:
        return self.value[0]

    def get_scrdir(self, fallback_scrdir: str = None) -> str:
        own_varname = self.get_scrdir_name()
        scrdir = os.environ.get(own_varname)
        if scrdir is None:
            for qmp in QMProgram:
                qmp_varname = qmp.get_scrdir_name()
                scrdir = os.environ.get(qmp_varname)
                if scrdir is not None:
                    eprint(f"WARNING: Failed to find environment variable {own_varname}; falling back to {qmp_varname} "
                           f"with value {scrdir}")
                    return scrdir
        else:
            return scrdir

        if scrdir is None:
            if fallback_scrdir is None:
                raise ValueError(f"ERROR: Could not find environment variable {own_varname} nor any fallback "
                                 f"environment variables.")
            else:
                eprint(f"WARNING: Could not find environment variable {own_varname} nor any fallback environment "
                       f"variables: falling back to provided scratch directory {fallback_scrdir}")
                return fallback_scrdir


def parse_qm_program(parsed: str, default_program: QMProgram = QMProgram.PSI4) -> QMProgram:
    parsed = parsed.upper()
    if parsed.startswith("GAUSS"):
        return QMProgram.GAUSSIAN
    elif parsed == "PSI4":
        return QMProgram.PSI4
    else:
        return default_program


def extract_molspec(qm_fi: str, program: QMProgram) -> (int, int, np.ndarray, np.ndarray):
    atoms = []
    coords = []
    charge = None
    spin = None
    if program == QMProgram.PSI4:
        with open(qm_fi, 'r') as r:
            line = r.readline()
            in_molspec = False
            while line != "":
                if in_molspec:
                    if "}" in line:
                        in_molspec = False
                    toks = line.strip().rstrip("}").rstrip().split()
                    if len(toks) > 3:
                        assert len(toks) < 6
                        atoms.append(toks[0])
                        coords.append(toks[1:4])
                    else:
                        assert not in_molspec
                elif line.startswith("molecule"):
                    if not line.rstrip().endswith("{"):
                        assert r.readline().strip() == "{"
                    toks = r.readline().strip().split()
                    assert len(toks) == 2
                    charge = int(toks[0])
                    spin = int(toks[1])
                    in_molspec = True
                line = r.readline()
    elif program == QMProgram.GAUSSIAN:
        with open(qm_fi, 'r') as r:
            line = r.readline()
            in_molspec = False
            after_routing = False
            while line != "":
                if in_molspec:
                    if line == "\n":
                        in_molspec = False
                        after_routing = False
                    else:
                        toks = line.strip().split()
                        assert len(toks) == 4
                        atoms.append(toks[0])
                        coords.append(toks[1:4])
                elif line.startswith("#"):
                    after_routing = True
                elif after_routing:
                    m = chargespin_patt.match(line)
                    if m:
                        toks = line.strip().split()
                        in_molspec = True
                        charge = int(toks[0])
                        spin = int(toks[1])
                line = r.readline()
    else:
        raise ValueError(f"Unrecognized QM program {program}")

    coords = np.array(coords, dtype=np.float64)
    assert charge is not None
    assert spin is not None
    assert len(atoms) > 0
    assert len(atoms) == coords.shape[0]
    assert coords.shape[1] == 3
    return charge, spin, np.array(atoms), coords


def list2_to_arr(list2: Sequence[list]) -> np.ndarray:
    return np.array([item for sublist in list2 for item in sublist])


gauss_float_patt = re.compile(r'-?0\.(\d+)D([+\-]\d{2})')


def convert_gauss_float(g_num: str) -> str:
    m = gauss_float_patt.match(g_num)
    if not m:
        raise ValueError(f"String {g_num} is not a Gaussian/Tinker style floating-point number!")
    neg_str = ""
    if g_num[0] == "-":
        neg_str = "-"

    nums = m.group(1)
    exp = int(m.group(2)) - 1
    return f"{neg_str}{nums[0]}.{nums[1:]}e{exp:d}"


def parse_gauss_float(g_num: str) -> float:
    return float(convert_gauss_float(g_num))


def symlink_nofail(source: str, dest: str) -> bool:
    """Creates a symbolic link if none already exists. Intended to avoid raising an error when the symlink already
    exists."""
    if os.path.exists(dest):
        return False
    os.symlink(source, dest)
    return True


singleton_patt = re.compile(r'^[0-9]+$')
range_patt = re.compile(r'^([0-9]+)-([0-9]+)$')


def parse_jml_range(range_str: str) -> Sequence[int]:
    """Adaptation of my comma-separated, 1-indexed, inclusive ranges from FFX."""
    toks = range_str.split(",")
    indices = set()
    for tok in toks:
        if singleton_patt.match(tok):
            indices.add(int(tok) - 1)
        else:
            m = range_patt.match(tok)
            if m:
                lb = int(m.group(1))
                ub = int(m.group(2))
                if (lb > ub):
                    eprint(f"Warning: specified range {tok} has lb > ub!")
                    temp = lb
                    lb = ub
                    ub = temp
                for i in range(lb - 1, ub, 1):
                    indices.add(i)
            else:
                eprint(f"Warning: unable to parse range {tok}")
    return sorted(indices)


def run_potential_3(potential: str, file_base: str):
    verbose_call([potential, "3", f"{file_base}.xyz", "Y"])


def parse_truth(val: str) -> bool:
    false_vals = ['false', '0', 'disabled', 'no']
    if val.lower() in false_vals:
        return False
    return bool(val)
