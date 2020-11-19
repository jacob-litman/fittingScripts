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
    return np.square(np.array(coord1) - np.array(coord2)).sum()


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
    assert charge is not None and spin is not None and len(atoms) > 0 and len(atoms) == coords.shape[0] and coords.shape[1] == 3
    return charge, spin, np.array(atoms), coords


def to_cartesian(x: np.ndarray, r: float) -> (np.ndarray, float, float, float, float):
    assert x.shape[0] == 2 and x.ndim == 1 and r > 0
    # Theta and phi
    t = x[0]
    p = x[1]
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    sin_p = math.sin(p)
    cos_p = math.cos(p)
    cart = np.array((r * sin_t * cos_p, r * sin_t * sin_p, r * cos_t))
    return cart, sin_t, cos_t, sin_p, cos_p
