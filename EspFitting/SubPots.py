#!/usr/bin/env python

import sys
import re
import math
import numpy as np
import io
from JMLUtils import eprint

DEFAULT_RADIUS = 2.5


def main():
    if not sys.argv or len(sys.argv) < 3:
        # TODO: Restore effective scriptiness (i.e. all arguments).
        eprint("Must have two arguments (potential files to add together).\n")
        sys.exit(1)
    main_inner(sys.argv[1], sys.argv[2])


def main_inner(f1: str, f2: str, out: io.TextIOWrapper = sys.stdout, err: io.TextIOWrapper = sys.stderr,
               subtract: bool = False, x: float = None, y: float = None, z: float = None, 
               radius: float = DEFAULT_RADIUS):
    all_diffs = []
    do_focus = False
    focus_xyz = (math.nan, math.nan, math.nan)
    focus_r2 = math.nan
    focus_diffs = []
    if x is not None:
        if subtract:
            focus_xyz = (x, y, z)
            focus_r2 = radius * radius
            do_focus = True
        else:
            err.write("Focus atom (region of special interest difference) is disabled for adding .pot files!\n")

    if subtract:
        err.write(f"Subtracting potential {f2} from {f1}\n")
    else:
        err.write(f"Adding potential {f2} to {f1}\n")
    keep_patt = re.compile(r"(.+ )(-?[0-9]+\.[0-9]+) *$")
    with open(f1, "r") as p1:
        with open(f2, "r") as psub:
            p1_line = p1.readline()
            psub_line = psub.readline()
            n_grid_points = int(p1_line.strip().split()[0])
            assert n_grid_points == int(psub_line.strip().split()[0])
            print(p1_line, end='')
            for i in range(n_grid_points):
                p1_line = p1.readline()
                p1_match = keep_patt.search(p1_line)
                psub_line = psub.readline()
                psub_match = keep_patt.search(psub_line)

                pot = float(p1_match.group(2))
                sub_pot = float(psub_match.group(2))
                if subtract:
                    this_diff = pot - sub_pot
                else:
                    this_diff = pot + sub_pot
                all_diffs.append(this_diff)
                out.write(f"{p1_match.group(1)}{this_diff:.6f}\n")
                if do_focus:
                    toks = p1_match.group(1).strip().split()
                    diff_xyz = (float(toks[1]) - focus_xyz[0], float(toks[2]) - focus_xyz[1], float(toks[3]) - focus_xyz[2])
                    dist2 = 0
                    for dx in diff_xyz:
                        dist2 += (dx * dx)
                    if dist2 <= focus_r2:
                        focus_diffs.append(this_diff)
                # TODO: Assertion that psub_line otherwise patches p1_line

    if subtract:
        all_arr = np.array(all_diffs)
        rms_all = math.sqrt(np.mean(np.square(all_arr)))
        mue_all = np.mean(np.abs(all_arr))
        mse_all = np.mean(all_arr)
        sd_all = np.std(all_arr, ddof=1)

        err.write(f"Statistics over all {len(all_diffs):d} grid points: RMSD {rms_all:.6f}, MUE {mue_all:.6f}, MSE {mse_all:.6f}, SD {sd_all:.6f}\n")
        if do_focus:
            all_arr = np.array(focus_diffs)
            rms_all = math.sqrt(np.mean(np.square(all_arr)))
            mue_all = np.mean(np.abs(all_arr))
            mse_all = np.mean(all_arr)
            sd_all = np.std(all_arr, ddof=1)
            err.write(f"Statistics over {len(focus_diffs):d} points in region of interest: RMSD {rms_all:.6f}, MUE {mue_all:.6f}, MSE {mse_all:.6f}, "
                   f"SD {sd_all:.6f}\n")
            err.write(f"Region of interest: {radius:.4f} Angstroms around {x:.4f},{y:.4f},{z:.4f}\n")


if __name__ == "__main__":
    main()
