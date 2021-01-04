import numpy as np
from typing import Sequence
from JMLUtils import eprint
import math
import scipy.optimize
from math import cos, sin


def unit_vector(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def vectors_cos(u: np.ndarray, v: np.ndarray) -> float:
    assert u.shape[0] == v.shape[0]
    assert u.ndim == 1 and v.ndim == 1
    assert np.count_nonzero(u) > 0 and np.count_nonzero(v) > 0
    u_u = unit_vector(u)
    v_u = unit_vector(v)
    return np.clip(np.dot(u_u, v_u), -1.0, 1.0)


def vectors_angle(u: np.ndarray, v: np.ndarray) -> float:
    return np.arccos(vectors_cos(u, v))


def project_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v * np.dot(u, v) / np.dot(v, v)


def reject_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u - project_vector(u, v)


def orthonormal_basis(x: np.ndarray, indices: Sequence[int] = None) -> np.ndarray:
    """From selected 3D Cartesian coordinates, produce an orthonormal basis set [i,j,k] where i is along the 0-1 axis, j
    is defined by the perpendicular component of 0-2, and k is defined by cross(i, j). If indices are not provided, use
    the first three sets of coordinates from x."""
    if indices is None:
        indices = [0, 1, 2]
    # Ensure three unique indices provided.
    assert len(indices) == 3
    assert len(set(indices)) == 3
    # Ensure indices are within the range of the coords array.
    assert min(indices) >= 0
    # Combined with above checks, ensures coords.shape[0] > 2.
    assert max(indices) < x.shape[0]
    # Ensure coords is of shape [n, 3]
    assert x.shape[1] == 3
    assert x.ndim == 2
    c0 = x[indices[0]]
    c1 = x[indices[1]]
    c2 = x[indices[2]]
    i = unit_vector(c1 - c0)
    d02 = c2 - c0
    assert abs(vectors_cos(i, d02)) < 1.0
    j = unit_vector(reject_vector(d02, i))
    k = np.cross(i, j)
    return np.vstack((i, j, k))


def spherical_cartesian(r: float, theta: float, phi: float) -> (float, float, float):
    sin_t = sin(theta)
    cos_t = cos(theta)
    sin_p = sin(phi)
    cos_p = cos(phi)
    return r * sin_t * cos_p, r * sin_t * sin_p, r * cos_t


def spherical_cartesian_derivs(r: float, theta: float, phi: float) -> np.ndarray:
    """Returns an array with columns for x, y, and z, rows for value and derivatives w.r.t. r, phi, theta."""
    values = np.empty((4, 3), dtype=np.float64)
    sin_t = sin(theta)
    cos_t = cos(theta)
    sin_p = sin(phi)
    cos_p = cos(phi)

    # r derivatives
    values[1][0] = sin_t * cos_p
    values[1][1] = sin_t * sin_p
    values[1][2] = cos_t
    # Cartesian coordinates
    values[0] = values[1] * r
    # Theta derivatives
    values[2][0] = r * cos_t * cos_p
    values[2][1] = r * cos_t * sin_p
    values[2][2] = -1 * r * sin_t
    # Phi derivatives
    values[3][0] = -1 * r * sin_t * sin_p
    values[3][1] = r * sin_t * cos_p
    values[3][2] = 0

    return values


def cartesian_spherical(x: float, y: float, z: float) -> (float, float, float):
    r = math.sqrt(x*x + y*y + z*z)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    # TODO: Implement cartesian_spherical_derivatives. The chain rule terms are unpleasant.
    return r, theta, phi


def plane_fit_costfun(x: np.ndarray, centroid: np.ndarray, points: np.ndarray) -> (float, np.ndarray):
    assert x.shape[0] == 2 and x.ndim == 1
    assert centroid.shape[0] == 3 and centroid.ndim == 1
    n_pts = points.shape[0]
    assert n_pts > 2 and points.shape[1] == 3 and points.ndim == 2
    e = 0
    values = spherical_cartesian_derivs(1.0, x[0], x[1])
    dedt = 0
    dedp = 0
    for i in range(n_pts):
        dxyz = points[i] - centroid
        e_pt = np.dot(dxyz, values[0])
        if e_pt < 0:
            sign = -1
        else:
            sign = 1
        e += sign * e_pt
        for j in range(3):
            dedt += (sign * values[2][j] * dxyz[j])
            dedp += (sign * values[3][j] * dxyz[j])
    return e, np.array([dedt, dedp], dtype=np.float64)


def best_fit_plane(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """Returns the centroid of a set of points plus the normal vector of its best-fit plane."""
    centroid = np.mean(points, axis=0, dtype=np.float64)
    opt_args = (centroid, points)
    # init_guess may actually be pretty good for many molecules on the xy plane!
    init_guess = np.array([0.0, 0.0], dtype=np.float64)
    bounds = [[-math.pi, math.pi], [-math.pi, math.pi]]
    options = {'maxiter': 500, 'disp': False, 'iprint': 0}
    eprint("Optimizing a best-fit plane.")
    opt_result = scipy.optimize.minimize(plane_fit_costfun, init_guess, args=opt_args, method='L-BFGS-B', jac=True,
                                         bounds=bounds, options=options)
    out_ijk = np.array(spherical_cartesian(1.0, opt_result.x[0], opt_result.x[1]))
    eprint(f"Resulting plane: passing through {centroid} with normal vector {out_ijk}")
    return centroid, out_ijk

def side_side_angle_triangle(b: float, c: float, beta: float, tol: float = 0.00001) -> np.ndarray:
    """Returns sides a, b, c and angles alpha, beta, and gamma given the side-side-angle solution. If there are
    non-unique solutions, the solution with obtuse gamma is returned as the second row. The tol parameter is used
    when calculating an asin: if value is in range 1 to (1 + tol), gamma assumed to be a right angle."""
    sin_g = (c / b) * math.sin(beta)
    degenerate_solution = False
    if sin_g > (1.0 + tol):
        raise ValueError("No valid SSA solution!")
    if sin_g >= 1.0:
        gamma = 0.5 * math.pi
    elif b >= c:
        gamma = math.asin(sin_g)
    else:
        degenerate_solution = True
        gamma = math.asin(sin_g)

    if degenerate_solution:
        alpha = math.pi - (beta + gamma)
        a = b * (sin(alpha) / sin(beta))
        ret_arr = np.empty((2, 6), dtype=np.float64)
        ret_arr[0] = [a, b, c, alpha, beta, gamma]
        # Now compute the obtuse solution.
        gamma = math.pi - gamma
        alpha = math.pi - (beta + gamma)
        a = b * (sin(alpha) / sin(beta))
        ret_arr[1] = [a, b, c, alpha, beta, gamma]
        return ret_arr
    else:
        alpha = math.pi - (beta + gamma)
        a = b * (sin(alpha) / sin(beta))
        return np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)
