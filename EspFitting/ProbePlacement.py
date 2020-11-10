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


def cost_gradient(x, weights: np.ndarray, x_all: np.ndarray, ind_center: int, r: float, exp: int = DEFAULT_EXP) -> (
        float, np.ndarray):
    assert weights.ndim == 1 and x_all.ndim == 2 and x.ndim == 1 and x.shape[0] == 2
    len_x = x_all.shape[0]
    assert len_x == weights.shape[0]

    (probe_cart, sin_t, cos_t, sin_p, cos_p) = to_cartesian(x, r)
    probe_cart = np.add(probe_cart, x_all[ind_center])

    dxdt = r * cos_t * cos_p
    dxdp = -1 * r * sin_t * sin_p
    dydt = r * cos_t * sin_p
    dydp = r * sin_t * cos_p
    dzdt = -1 * r * sin_t
    # dzdp == 0

    tot_penalty = 0
    grad = np.zeros_like(x)
    for i in range(len_x):
        if i == ind_center:
            continue
        xi = x_all[i]
        d2 = dist2(probe_cart, xi)
        tot_penalty += (weights[i] / d2 ** exp)
        const = -1 * exp * weights[i] / (d2 ** (exp + 1))

        d0 = probe_cart[0] - xi[0]
        d1 = probe_cart[1] - xi[1]
        d2 = probe_cart[2] - xi[2]

        dmdt = 2 * d0 * dxdt
        dmdt += 2 * d1 * dydt
        dmdt += 2 * d2 * dzdt
        grad[0] += (dmdt * const)
        dmdp = 2 * d0 * dxdp
        dmdp += 2 * d1 * dydp
        # dzdp == 0, so that whole term is zero.
        grad[1] += (dmdp * const)
    return tot_penalty, grad


def calc_drdk(dists: np.ndarray, unitvec: np.ndarray, r: float) -> float:
    drdk = 0
    for j in range(3):
        drdk += (dists[j] * unitvec[j])
    drdk /= r
    return drdk


def linear_costgrad(x, weights: np.ndarray, x_all: np.ndarray, ind_center: int, unitvec: np.ndarray,
                    exp: int = 1, min_bound: float = 4.0, max_bound: float = 5.0) -> (float, np.ndarray):
    assert weights.ndim == 1 and x_all.ndim == 2 and x.ndim == 1 and x.shape[0] == 2
    assert len(unitvec) == 3
    len_x = x_all.shape[0]
    assert len_x == weights.shape[0]
    k = x[0]
    # Energy and derivatives w.r.t. k
    ek = 0
    dedk = 0
    # Exponent minus one
    e1 = exp - 1

    probe_cart = k * unitvec
    probe_cart += x_all[ind_center]
    min_bound_2 = min_bound * min_bound
    max_bound_2 = max_bound * max_bound

    for i in range(len_x):
        xi = x_all[i]
        dists = xi - probe_cart
        dists2 = np.square(dists)
        r2 = np.sum(dists2)
        if r2 < min_bound_2:
            r = math.sqrt(r2)
            dbound = min_bound - r
            ek += weights[i] * (dbound ** exp)
            drdk = calc_drdk(dists=dists, unitvec=unitvec, r=r)
            dedk -= weights[i] * exp * (dbound ** e1) * drdk
        elif i == ind_center and r2 > max_bound_2:
            r = math.sqrt(r2)
            dbound = r - max_bound
            ek += weights[i] * (dbound ** exp)
            drdk = calc_drdk(dists=dists, unitvec=unitvec, r=r)
            dedk += weights[i] * (dbound ** e1) * drdk
        # Else, no energy/derivative at the flat bottom.

    return ek, np.array([dedk])


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
    parser.add_argument('-l', dest='linopt', type=bool, action='store_true', help='After determining the direction where '
                                                                                  'the probe should be placed, additionally relax the distance.')
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


def inner_loop(xyz_input: StructXYZ, ai: int, dx: float, exponent: int, real_xyz: np.ndarray, avoid_weight: np.ndarray,
               out_file_base: str, xyzpdb: str, out_locs: np.ndarray, keyf: str = None):
    if keyf is None:
        keyf = xyz_input.key_file
    center_xyz = real_xyz[ai]
    n_ats = real_xyz.shape[0]
    atom_quick_id = f"{xyz_input.atom_names[ai]}{ai + 1:d}"

    pi2 = 0.5 * pi
    guesses = np.array([[0, 0], [pi2, 0], [pi2, pi2], [pi2, pi], [pi2, -1 * pi2], [pi, 0]])

    opt_args = (avoid_weight, real_xyz, ai, dx, exponent)
    outs = []
    out_carts = []
    vector_carts = []
    lowest_cart = None
    lowest_e = np.finfo(float).max
    method = 'BFGS'
    linear_method='L-BFGS-B'
    eprint(f"Beginning {method} maximization of distance from atom {atom_quick_id} to non-target atoms.")
    eprint(f"Atomic center: {center_xyz}")

    ctr = 1
    max_dx = dx + 1.0
    linear_bounds = [[dx, max_dx + 1.0]]

    for guess in guesses:
        eprint(f"Starting from guess {guess} w/ Cartesian coordinates {np.add(to_cartesian(guess, dx)[0], center_xyz)}")
        opt_result = scipy.optimize.minimize(cost_gradient, guess, args=opt_args, method=method, jac=True,
                                             options={'disp': False, 'maxiter': 10000, 'gtol': 1E-9})
        if not opt_result.success:
            eprint(f'\nWARNING: Directional optimization was not a success! Status: {opt_result.status}')
            eprint(f'Error message: {opt_result.message}\n')
        this_out = opt_result.x
        this_cart = to_cartesian(this_out, dx)[0]

        # TODO: Set min_bound and max_bound
        linopt_args = (avoid_weight, real_xyz, ai, this_cart, exponent, dx, max_dx)
        linopt_result = scipy.optimize.minimize(linear_costgrad, np.array([dx]), args=linopt_args, method=linear_method,
                                                jac=True, bounds=linear_bounds,
                                                options={'disp': True, 'maxiter': 1000, 'gtol': 1E-9})
        if not linopt_result.success:
            eprint(f'\nWARNING: Distance optimization was not a success! Status: {opt_result.status}')
            eprint(f'Error message: {opt_result.message}\n')
        e_orig = linopt_result.fun
        k_orig = linopt_result.x[0]
        min_e = e_orig
        min_k = linopt_result.x
        for i in range(10):
            test_k = k_orig + (0.01 * i)
            test_arr = np.array([test_k])
            test_e = linear_costgrad(test_arr, avoid_weight, real_xyz, ai, this_cart, exponent, dx, max_dx)
            if test_e < min_e:
                eprint(f"Found energy {test_e:.5f} < optimization result {e_orig:.5f} at separation {test_k} Angstroms")
                min_e = test_e
                min_k = test_k


        this_cart += center_xyz
        vector_carts.append(this_cart)
        out_carts.append(this_cart)
        this_out = np.rad2deg(this_out)
        outs.append(this_out)
        eprint(f'Energy after {opt_result.nit} iterations: {opt_result.fun:.5g}')
        if opt_result.fun < lowest_e:
            lowest_e = opt_result.fun
            lowest_cart = this_cart
        eprint(f'Output coordinates: theta {this_out[0]:.5f}, phi {this_out[1]:.5f}, r , XYZ {this_cart}\n')
        xyz_input.coords[n_ats, :] = this_cart
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
