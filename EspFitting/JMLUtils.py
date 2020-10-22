import sys
import os
import subprocess
import re
from typing import Sequence, Dict, Mapping

cryst_patt = re.compile(r"^ +(-?\d+\.\d+){6} *$")


def eprint(*args, kwargs: dict = None):
    if kwargs is None:
        kwargs = dict()
    kwargs['file'] = sys.stderr
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
