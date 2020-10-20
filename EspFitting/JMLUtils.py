import sys
import os
import subprocess
import re
from typing import Sequence

cryst_patt = re.compile(r"^ +(-?\d+\.\d+){6} *$")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
