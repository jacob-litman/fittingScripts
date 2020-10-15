import sys
import os


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def version_file(fname: str) -> str:
    if os.path.exists(fname):
        for i in range(1000):
            trial_fname = f"{fname}_{i}"
            if not os.path.exists(trial_fname):
                return trial_fname
        raise RuntimeError(f"Could not version {fname} to {fname}_x for 1<=x<=1000!")
    else:
        return fname
