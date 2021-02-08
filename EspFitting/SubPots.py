#!/usr/bin/env python

import sys
import re
import math
import numpy as np
import io
from JMLUtils import eprint
import scipy.stats

DEFAULT_RADIUS = 2.5


def main():
    if not sys.argv or len(sys.argv) < 3:
        # TODO: Restore effective scriptiness (i.e. all arguments).
        eprint("Must have two arguments (potential files to add together).\n")
        sys.exit(1)
    else:
        main_inner(sys.argv[1], sys.argv[2], subtract=True)


def main_inner(f1: str, f2: str, out: io.TextIOWrapper = sys.stdout, err: io.TextIOWrapper = sys.stderr,
               subtract: bool = False, x: float = None, y: float = None, z: float = None,
               radius: float = DEFAULT_RADIUS, header_comment: str = None):
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

    if header_comment is None:
        if subtract:
            header_comment = 'Subtraction.pot'
        else:
            header_comment = 'Addition.pot'

    n_grid_points = -1
    p1_header = False
    with open(f1, "r") as p1:
        line = p1.readline()
        if line.strip().split()[0] == "1":
            #eprint(f"File {f1} has no header.")
            num_lines = 1
        else:
            num_lines = 0
            n_grid_points = int(line.strip().split()[0])
            p1_header = True

        while p1.readline().strip() != "":
            num_lines += 1

        if p1_header:
            assert num_lines == n_grid_points
        else:
            n_grid_points = num_lines

    p2_header = False
    with open(f2, "r") as p2:
        line = p2.readline()
        if line.strip().split()[0] == "1":
            #eprint(f"File {f2} has no header.")
            num_lines = 1
        else:
            num_lines = 0
            p2_header = True

        while p2.readline().strip() != "":
            num_lines += 1
        assert num_lines == n_grid_points

    with open(f1, "r") as p1:
        with open(f2, "r") as p2:
            if p1_header:
                p1.readline()
            if p2_header:
                p2.readline()
            out.write(f" {n_grid_points:>7d}   {header_comment}\n")
            for i in range(n_grid_points):
                p1_line = p1.readline()
                p1_match = keep_patt.search(p1_line)
                psub_line = p2.readline()
                psub_match = keep_patt.search(psub_line)

                pot = float(p1_match.group(2))
                sub_pot = float(psub_match.group(2))
                if subtract:
                    this_diff = pot - sub_pot
                    all_diffs.append(this_diff)
                else:
                    this_diff = pot + sub_pot
                out.write(f"{p1_match.group(1)}{this_diff:.6f}\n")
                if do_focus:
                    toks = p1_match.group(1).strip().split()
                    diff_xyz = (float(toks[1]) - focus_xyz[0], float(toks[2]) - focus_xyz[1],
                                float(toks[3]) - focus_xyz[2])
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

        err.write(f"Statistics over all {len(all_diffs):d} grid points: RMSD {rms_all:.6f}, MUE {mue_all:.6f}, "
                  f"MSE {mse_all:.6f}, SD {sd_all:.6f}\n")
        if do_focus and len(focus_diffs) > 0:
            all_arr = np.array(focus_diffs)
            rms_all = math.sqrt(np.mean(np.square(all_arr)))
            mue_all = np.mean(np.abs(all_arr))
            mse_all = np.mean(all_arr)
            sd_all = np.std(all_arr, ddof=1)
            err.write(f"Statistics over {len(focus_diffs):d} points in region of interest: RMSD {rms_all:.6f}, "
                      f"MUE {mue_all:.6f}, MSE {mse_all:.6f}, SD {sd_all:.6f}\n")
            err.write(f"Region of interest: {radius:.4f} Angstroms around {x:.4f},{y:.4f},{z:.4f}\n")


if __name__ == "__main__":
    main()
