import argparse
import math
import os
import re
import subprocess
from math import pi

import numpy as np
import scipy.optimize

from JMLUtils import eprint, dist2
from StructureXYZ import StructXYZ

DEFAULT_HWT = 0.4
DEFAULT_DIST = 4.0
DEFAULT_OUTFILE = 'probe'
DEFAULT_PROBE_DESC = "Probe Charge        "
DEFAULT_EXP = 3
DEFAULT_PROBE_TYPE = 999


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


# Following would be variables to cache the Hessian.
#last_x = None
#last_hess = None


def cost_jac_inner(negative: bool, rm2: float, bound_dist: float, i: int, weights: np.ndarray, exp: int,
               rv: np.ndarray) -> (float, np.ndarray):
    r = math.sqrt(rm2)    
    if negative:
        pen_dist = bound_dist - r
    else:
        pen_dist = r - bound_dist
    e = weights[i] * (pen_dist ** exp)
    
    # Compute gradient.
    exp1 = exp - 1
    r_inv = 1.0 / r
    grad = np.empty(3, float)
    r_grad = np.empty_like(grad)
    const_grad = weights[i] * exp

    # Commented code is only needed for computing the Hessian, but I'm not sure there's an elegant way to get Scipy to
    # use one method call for value, Jacobian and Hessian.
    """# exp2 and const_lhs only needed for Hessian: computed here to have only one if-negative branch.
    exp2 = exp - 2
    const_lhs = exp1 * (pen_dist ** exp2)"""
    if negative:
        const_grad *= -1
        # const_lhs *= -1
        
    lhs_grad = pen_dist ** exp1
    for j in range(3):
        r_grad[j] = rv[j] * r_inv
        grad[j] = const_grad * lhs_grad * r_grad[j]

    """# Compute the Hessian.
    hess = np.empty((3, 3), float)
    r_hess = np.empty_like(hess)
    r_inv3 = r_inv * r_inv * r_inv
    for j in range(3):
        for k in range(3):
            r_hess[j][k] = -1 * rv[j] * rv[k] * r_inv3
            if j == k:
                r_hess[j][k] += r_inv
            hess_jk = const_lhs * r_grad[j] * r_grad[k]
            hess_jk += lhs_grad * r_hess[j][k]
            hess_jk *= const_grad
            hess[j][k] = hess_jk
    return e, grad, hess"""
    return e, grad


def cost_jac_hess(x, weights: np.ndarray, x_all: np.ndarray, ind_center: int, exp: int = DEFAULT_EXP,
                  min_dist: float = 4.0, pen_dist: float = 5.0) -> (float, np.ndarray, np.ndarray):
    assert weights.ndim == 1 and x_all.ndim == 2 and x.ndim == 1 and x.shape[0] == 3
    assert 0 < min_dist < pen_dist
    len_x = x_all = x_all.shape[0]
    assert len_x == weights.shape[0]
    
    min_dist2 = min_dist * min_dist
    max_dist2 = pen_dist * pen_dist
    
    e = 0
    grad = np.zeros(3, float)
    # Hessian-related values commented out.
    #hess = np.zeros((3, 3), float)
    
    for i in range(len_x):
        xi = x_all[i]
        # rv is distance vector.
        rv = x - xi
        rv2 = np.square(rv)
        # rm2 is the square of the distance magnitude
        rm2 = np.sum(rv2)
        if rm2 < min_dist2:
            result = cost_jac_inner(True, rm2, min_dist, i, weights, exp, rv)
            e += result[0]
            grad += result[1]
            #hess += result[2]
        elif i == ind_center and rm2 > max_dist2:
            result = cost_jac_inner(False, rm2, min_dist, i, weights, exp, rv)
            e += result[0]
            grad += result[1]
            #hess += result[2]
    global last_x, last_hess
    last_x = x.copy()
    #last_hess = hess.copy()
    return e, grad
    

"""def get_hess(x, *args) -> np.ndarray:
    global last_x, last_hess
    if isinstance(last_x, np.ndarray) and np.array_equal(x, last_x) and isinstance(last_hess, np.ndarray):
        return last_hess.copy()
    else:
        last_x = x
        last_hess = cost_jac_hess(x, *args)[2]
        return last_hess.copy()"""
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='probe_name', type=str, default='PC', help='Atom name to give the probe')
    parser.add_argument('-t', dest='probe_atype', type=int, default=DEFAULT_PROBE_TYPE, help='Atom type to assign to '
                                                                                             'the probe')
    parser.add_argument('-d', dest='distance', type=float, default=DEFAULT_DIST, help='Distance to place the probe at')
    parser.add_argument('-w', dest='hydrogen_weight', type=float, default=DEFAULT_HWT, help='Relative weighting for '
                                                                                    'hydrogen distances')
    parser.add_argument('-e', dest='exp', type=int, default=DEFAULT_EXP, help='Exponent for the square of distance in '
                                                                              'the target function')
    parser.add_argument('-x', dest='xyzpdb', type=str, default='xyzpdb', help='Name or full path of Tinker xyzpdb')
    parser.add_argument('infile', nargs=1, type=str)

    args = parser.parse_args()

    xyz_input = StructXYZ(args.infile[0], args.keyfile)

    main_inner(xyz_input, args.distance, args.hydrogen_weight, args.exp, args.probe_atype, xyzpdb=args.xyzpdb)


def main_inner(xyz_input: StructXYZ, at: int = DEFAULT_PROBE_TYPE, dx: float = DEFAULT_DIST,
               exponent: int = DEFAULT_EXP, hwt: float = DEFAULT_HWT, out_file_base: str = DEFAULT_OUTFILE,
               xyzpdb: str = 'xyzpdb', probe_type = None, keyf: str = None) -> np.ndarray:
    """Main driver for the script; can be called externally as well. Returns the atom type used for the probe."""
    n_real_ats = xyz_input.n_atoms
    real_xyz = xyz_input.coords.copy()
    avoid_weight = np.ones(n_real_ats)
    anames = xyz_input.atom_names
    for i in range(n_real_ats):
        if anames[i][0] == 'H':
            avoid_weight[i] = hwt

    # Magic Variables Here (TODO: configure)
    probe_anum = 999
    probe_mass = 1.0
    '''if at not in xyz_input.probe_types:
        xyz_input.append_atype_def(at, at, 'PC', DEFAULT_PROBE_DESC, probe_anum, probe_mass, 0, isprobe=True)'''
    if probe_type is None:
        probe_type = xyz_input.append_atype_def(at, at, 'PC', DEFAULT_PROBE_DESC, probe_anum, probe_mass, 0,
                                                isprobe=True)
        at = probe_type[0]
    xyz_input.append_atom('PC', np.zeros(3), at)

    probe_locs = np.zeros((n_real_ats, 3))

    for i in range(n_real_ats):
        inner_loop(xyz_input, i, dx, exponent, real_xyz, avoid_weight, out_file_base, xyzpdb, probe_locs, keyf)

    return probe_locs


def check_valid_probe(probe_xyz: np.ndarray, real_xyz: np.ndarray, min_dx2: float) -> bool:
    for i in range(len(real_xyz)):
        xyzi = real_xyz[i]
        dx2 = dist2(xyzi, probe_xyz)
        if dx2 < min_dx2:
            return False
    return True


def inner_loop(xyz_input: StructXYZ, ai: int, dx: float, exponent: int, real_xyz: np.ndarray, avoid_weight: np.ndarray,
               out_file_base: str, xyzpdb: str, out_locs: np.ndarray, keyf: str = None, pen_ddx: float = 1.0,
               bound_ddx: float = 2.0):
    assert 0 < pen_ddx < bound_ddx and dx > 0 and exponent > 0

    pen_ddx += dx
    bound_ddx += dx
    if keyf is None:
        keyf = xyz_input.key_file
    center_xyz = real_xyz[ai]
    n_ats = real_xyz.shape[0]
    atom_quick_id = f"{xyz_input.atom_names[ai]}{ai + 1:d}"

    guesses = np.empty((6, 3), dtype=float)
    for i in range(6):
        guesses[i] = center_xyz.copy()
    guesses[0][0] += dx
    guesses[1][0] -= dx
    guesses[2][1] += dx
    guesses[3][1] -= dx
    guesses[4][2] += dx
    guesses[5][2] -= dx
    
    bounds = np.zeros((3, 2))
    for i in range(3):
        bounds[i][0] = center_xyz[i] - bound_ddx
        bounds[i][1] = center_xyz[i] + bound_ddx

    opt_args = (avoid_weight, real_xyz, ai, exponent, dx, pen_ddx)
    out_carts = []
    vector_carts = []
    lowest_cart = None
    lowest_e = np.finfo(float).max
    method = 'L-BFGS-B'
    eprint(f"Beginning {method} maximization of distance from atom {atom_quick_id} to non-target atoms.")
    eprint(f"Atomic center: {center_xyz}")

    ctr = 1

    for guess in guesses:
        eprint(f"Starting from guess {guess} w/ Cartesian coordinates {np.add(to_cartesian(guess, dx)[0], center_xyz)}")
        opt_result = scipy.optimize.minimize(cost_jac_hess, guess, args=opt_args, method=method, jac=True,
                                             options={'disp': True, 'maxiter': 10000, 'gtol': 1E-9})

        if not opt_result.success:
            eprint(f'\nWARNING: Optimization was not a success! Status: {opt_result.status}')
            eprint(f'Error message: {opt_result.message}\n')
        #this_out = opt_result.x
        #this_cart = to_cartesian(this_out, dx)[0]
        this_cart = opt_result.x

        vector_carts.append(this_cart)
        out_carts.append(this_cart)
        eprint(f'Energy after {opt_result.nit} iterations: {opt_result.fun:.5g}')
        if opt_result.fun < lowest_e:
            lowest_e = opt_result.fun
            lowest_cart = this_cart
        eprint(f'Output coordinates: {this_cart[0]:.5f}, {this_cart[1]:.5f}, {this_cart[2]:.5f}')
        ctr += 1
        xyz_input.coords[n_ats, :] = center_xyz

    assert lowest_cart is not None
    os.mkdir(atom_quick_id)
    xyz_input.coords[n_ats, :] = lowest_cart
    outf = f"{atom_quick_id}{os.sep}{out_file_base}.xyz"
    xyz_input.write_out(outf)

    for i in range(3):
        out_locs[ai, :] = lowest_cart

    if keyf is not None:
        # Append .pdb afterwards just in case of weird filename shenaniganry.
        cmdstr = f"{xyzpdb} {outf} {keyf}\n"
        eprint(f"Calling {cmdstr}")
        subprocess.run([xyzpdb, outf, keyf])


if __name__ == "__main__":
    main()
