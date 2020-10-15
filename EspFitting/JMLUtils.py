import sys
import os
import subprocess
from typing import List


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def verbose_call(args: List[str], to_stderr: bool = True, **kwargs):
    out_str = "Calling:"
    for a in args:
        out_str += f" {a}"
    if to_stderr:
        eprint(out_str)
    else:
        print(out_str)
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
