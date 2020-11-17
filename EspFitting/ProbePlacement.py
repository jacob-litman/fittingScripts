import argparse
import math
import os
import subprocess

import numpy as np
import scipy.optimize

from JMLUtils import eprint
from StructureXYZ import StructXYZ

DEFAULT_HWT = 0.4
DEFAULT_DIST = 4.0
DEFAULT_OUTFILE = 'probe'
DEFAULT_EXP = 6
DEFAULT_MIN_DIST = 4.0
DEFAULT_RESTRAIN_DIST = 4.2
DEFAULT_MAX_DIST = 6.0
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='probe_atype', type=int, default=None,
                        help='Pre-existing atom type (in key file) to assign; else a default probe type is appended.')
    parser.add_argument('-d', dest='distance', type=float, default=DEFAULT_DIST,
                        help='Strongly repel probe from other atoms within this distance')
    parser.add_argument('-r', dest='restrain_distance', type=float, default=DEFAULT_RESTRAIN_DIST,
                        help='Restrain probe to be within this distance of the probed atom.')
    parser.add_argument('-m', dest='max_distance', type=float, default=DEFAULT_MAX_DIST,
                        help='Maximum distance to the probe (hard constraint)')
    parser.add_argument('-w', dest='hydrogen_weight', type=float, default=DEFAULT_HWT,
                        help='Relative weighting for hydrogen distances')
    parser.add_argument('-e', dest='exponent', type=int, default=DEFAULT_EXP,
                        help='Exponent for the square of distance in the target function')
    parser.add_argument('-x', dest='xyzpdb', type=str, default='xyzpdb', help='Name or full path of Tinker xyzpdb')
    parser.add_argument('--inKey', dest='keyfile', type=str, default=None,
                        help='Name of input .key file (else from infile)')
    parser.add_argument('--fullhelp', dest='fullhelp', action='store_true',
                        help='Fully describe the behavior of ProbePlacement; requires a dummy infile argument..')
    parser.add_argument('infile', nargs=1, type=str)

    args = parser.parse_args()

    if args.fullhelp:
        eprint("This script attempts to place one probe per atom in the specified molecule in such a way as to satisfy "
               "these conditions:")
        eprint(f"#1: The probe should be no closer than {args.distance:.2f} (-d) Angstroms to any solute atom.")
        eprint(f"#2: The probe must be no further than {args.max_distance:.2f} (-m) Angstroms from the targeted atom.")
        eprint(f"#3: If possible, the probe should be no further than {args.restrain_distance:.2f} (-r) Angstroms from "
               f"the targeted atom.")
        eprint(f"#4: The probe should be as far as possible from non-target atoms while satisfying #2-3.")
        eprint(f"#5: Hydrogens should be 'less important', and their contributions are downweighted to a factor of "
               f"{args.hydrogen_weight} (-w) in all components of the target function.")

        eprint("\nAs such, the script uses six test placements, initially placed octahedrally around the test atom.")
        eprint("These are then locally optimized to a target function to find six local minima.")
        eprint("These are then extended out in steps of 0.01A (up to 10x) in case the local optimizer terminated "
               "early.")
        eprint("The placement with the lowest value of the target function and no violations of #1 is kept; if there"
               "are no placements without a constraint violation, a warning is emitted.")
        eprint(f"In all cases, the distance to the target atom is constrained to the range "
               f"{args.distance:.2f}-{args.max_distance:.2f} (-d to -m) Angstroms.")

        eprint("\nThe target function is split into three parts, where r is distance from probe to a solute atom.")
        eprint("To avoid confusion: the exponent (-e) is renamed to n, and the restraint distance (-r) is renamed to a")
        eprint("All other variables are their command-line names (e.g. d is the distance argument -d)")

        eprint("\nThe 'strong' function is targeted at condition #1; E(strong) = (d - r)**n for 0 < r < d")
        eprint("The probe is constrained to distances d <= r <= m from the target atom.")
        eprint("The strong component is 0 for r >= d")
        eprint("The target-restraint component (incorporated into strong-component calculations) is targeted at #2-3.")
        eprint(f"E(tr) = (r - a)**e for a < r <= m")
        eprint("The weak component is targeted at condition #4; E(weak) = r**(-e) for all r.\n\n")
        parser.print_help()
    else:
        xyz_input = StructXYZ(args.infile[0], args.keyfile)
        # TODO: Non-default out_file_base.
        main_inner(xyz_input, probe_type=args.probe_atype, min_dist=args.distance, restrain_dist=args.restrain_distance,
                   max_dist=args.max_distance, exp=args.exponent, hwt=args.hydrogen_weight, xyzpdb=args.xyzpdb)


def main_inner(xyz_input: StructXYZ, keyf: str = None, out_file_base: str = DEFAULT_OUTFILE, probe_type = None,
               min_dist: float = DEFAULT_MIN_DIST, restrain_dist: float = DEFAULT_RESTRAIN_DIST,
               max_dist: float = DEFAULT_MAX_DIST, exp: int = DEFAULT_EXP, hwt: float = DEFAULT_HWT,
               weight_weak: float = 1.0, xyzpdb: str = 'xyzpdb'):
    assert len(xyz_input.probe_indices) == 0
    if probe_type is None:
        probe_type = xyz_input.get_default_probetype()[0]
    elif isinstance(probe_type, tuple):
        probe_type = probe_type[0]
    else:
        assert isinstance(probe_type, int)
    assert probe_type in xyz_input.probe_types

    """Main driver for the script; can be called externally as well. Returns the atom type used for the probe."""
    n_real_ats = xyz_input.n_atoms
    real_xyz = xyz_input.coords.copy()
    avoid_weight = np.ones(n_real_ats)
    anames = xyz_input.atom_names
    for i in range(n_real_ats):
        if anames[i][0] == 'H':
            avoid_weight[i] = hwt

    xyz_input.append_atom(probe_type, np.zeros(3))

    probe_locs = np.zeros((n_real_ats, 3))

    for i in range(n_real_ats):
        inner_loop(xyz_input, i, real_xyz, avoid_weight, probe_type, keyf, out_file_base, min_dist, restrain_dist,
                   max_dist, exp, weight_weak, xyzpdb)

    return probe_locs


def inner_loop(xyz_input: StructXYZ, ai: int, real_xyz: np.ndarray, avoid_weight: np.ndarray, probe_type: int,
               keyf: str = None, out_file_base: str = DEFAULT_OUTFILE, min_dist: float = DEFAULT_MIN_DIST,
               restrain_dist: float = DEFAULT_RESTRAIN_DIST, max_dist: float = DEFAULT_MAX_DIST, exp: int = DEFAULT_EXP,
               weight_weak: float = 1.0, xyzpdb: str = 'xyzpdb'):
    assert 0 < min_dist <= restrain_dist < max_dist
    assert probe_type in xyz_input.probe_types
    assert exp > 1
    assert weight_weak >= 0
    assert len(xyz_input.probe_indices) == 1

    if keyf is None:
        keyf = xyz_input.key_file
    center_xyz = real_xyz[ai].copy()
    n_ats = real_xyz.shape[0]
    atom_quick_id = f"{xyz_input.atom_names[ai]}{ai + 1:d}"

    guesses = np.empty((6, 3), dtype=float)
    for i in range(6):
        guesses[i] = center_xyz.copy()
    guesses[0][0] += min_dist
    guesses[1][0] -= min_dist
    guesses[2][1] += min_dist
    guesses[3][1] -= min_dist
    guesses[4][2] += min_dist
    guesses[5][2] -= min_dist
    
    bounds = np.zeros((3, 2))
    for i in range(3):
        bounds[i][0] = center_xyz[i] - max_dist
        bounds[i][1] = center_xyz[i] + max_dist

    opt_args = (avoid_weight, real_xyz, ai, exp, min_dist, restrain_dist, weight_weak)
    out_positions = []
    lowest_position = []
    lowest_e = np.finfo(float).max
    method = 'L-BFGS-B'
    eprint(f"Placing probe near atom {atom_quick_id}.")
    eprint(f"Atomic center: {center_xyz}")

    ctr = 1

    for guess in guesses:
        #eprint(f"Starting from guess {guess} w/ Cartesian coordinates {np.add(to_cartesian(guess, dx)[0], center_xyz)}")
        eprint(f"Starting from guess {guess}")
        opt_result = scipy.optimize.minimize(cost_jac, guess, args=opt_args, method=method, jac=True,
                                             options={'disp': True, 'maxiter': 10000, 'gtol': 1E-9}, bounds=bounds)
        if not opt_result.success:
            eprint(f'\nWARNING: Optimization was not a success! Status: {opt_result.status}')
            eprint(f'Error message: {opt_result.message}\n')
        x = opt_result.x
        eprint(f"Initial result of optimization: {x}")
        init_fun = opt_result.fun
        eprint(f'Energy after {opt_result.nit} iterations: {init_fun:.5g}')
        extend_delta = 1E-4
        extend = extend_guess(center_xyz, x, opt_args, max_dist=max_dist, delta=extend_delta)
        assert extend[1][0] < init_fun or np.isclose(extend[1][0], init_fun)

        if extend[0] > 0:
            extend_dist = extend[0] * extend_delta
            e_reduction = init_fun - extend[1][0]
            eprint(f"Probe placement extended by {extend_dist:.4g} Angstroms w/ energy reduction {e_reduction:.4g}")

        out_positions.append(extend[2])
        if extend[1][0] < lowest_e:
            lowest_e = extend[1][0]
            lowest_position = extend[2]
        eprint(f'Output coordinates: {extend[2][0]:.5f}, {extend[2][1]:.5f}, {extend[2][2]:.5f}')
        ctr += 1
        # Reset probe coordinates.
        xyz_input.coords[n_ats, :] = center_xyz

    assert lowest_position is not None
    os.mkdir(atom_quick_id)
    xyz_input.coords[n_ats, :] = lowest_position
    outf = f"{atom_quick_id}{os.sep}{out_file_base}.xyz"
    xyz_input.write_out(outf)

    if keyf is not None and xyzpdb is not None:
        # Append .pdb afterwards just in case of weird filename shenaniganry.
        cmdstr = f"{xyzpdb} {outf} {keyf}\n"
        eprint(f"Calling {cmdstr}")
        subprocess.run([xyzpdb, outf, keyf])


def cost_jac(x, weights: np.ndarray, x_all: np.ndarray, index_center: int, exp: int, min_dist: float,
             restrain_dist: float, weight_weak: float) -> (float, np.ndarray):
    assert weights.ndim == 1 and x_all.ndim == 2 and x.ndim == 1 and x.shape[0] == 3
    assert 0 < min_dist < restrain_dist
    len_x = x_all.shape[0]
    assert len_x == weights.shape[0]

    e_weak = 0
    e_strong = 0
    grad = np.zeros(3, np.float64)
    exp1 = exp - 1
    inv_exp = -1 * exp
    inv_exp1 = inv_exp - 1

    for i in range(len_x):
        xi = x_all[i]
        # rv is distance vector.
        rv = x - xi
        rv2 = np.square(rv)
        # rm2 is the square of the distance magnitude
        rm2 = float(np.sum(rv2))
        # r is the distance magnitude.
        r = math.sqrt(rm2)
        r_inv = 1.0 / r

        e_weak += weights[i] * weight_weak * (r ** inv_exp)

        # Gradient of distance w.r.t. x,y,z
        r_grad = np.empty(3, np.float64)
        gradient_prefix = weights[i] * exp
        for j in range(3):
            r_grad[j] = rv[j] * r_inv
            grad[j] -= gradient_prefix * (r ** inv_exp1) * r_grad[j]

        do_strong = False
        pen_stretch = 0
        if r < min_dist:
            do_strong = True
            pen_stretch = min_dist - r
            gradient_prefix *= -1
        elif r > restrain_dist and i == index_center:
            do_strong = True
            pen_stretch = r - restrain_dist

        if do_strong:
            e_strong += weights[i] * (pen_stretch ** exp)
            lhs_grad = (pen_stretch ** exp1) * gradient_prefix
            for j in range(3):
                grad[j] += lhs_grad * r_grad[j]

    e_tot = e_weak + e_strong
    return e_tot, grad


def extend_guess(center_xyz: np.ndarray, init_guess: np.ndarray, cost_args: tuple, delta: float = 1E-4,
                 maxiter: int = 20, max_dist: float = DEFAULT_MAX_DIST) -> (int, float, np.ndarray):
    vec = init_guess - center_xyz
    mag = math.sqrt(np.dot(vec, vec))
    vec /= mag
    vec *= delta

    min_e = cost_jac(init_guess, *cost_args)
    min_x = init_guess.copy()
    min_index = 0
    for i in range(1, maxiter + 1, 1):
        if (mag + (i * delta)) >= max_dist:
            break
        guess = init_guess + (i * vec)
        e = cost_jac(guess, *cost_args)
        if e < min_e:
            min_e = e
            min_x = guess.copy()
            min_index = i
    return min_index, min_e, min_x


if __name__ == "__main__":
    main()
