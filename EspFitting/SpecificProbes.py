import argparse
import ProbePlacers.Ammonia
import ProbePlacers.Water
import ProbePlacers.Benzene
import sys
from JMLUtils import eprint, dist2
import numpy as np
import JMLMath
import math
from StructureXYZ import StructXYZ
from typing import Sequence


def place_ethene(infile: str, delta=4.0):
    delta = float(delta)
    xyzfi = StructXYZ(infile)
    ProbePlacers.Water.place_triangle(xyzfi, outname="ETHENE-PROBE-1.xyz", center=0, flank1=2, flank2=3)
    probe_index = xyzfi.n_atoms - 1

    """Define x as the 1-0 vector (carbon-carbon bond), y as reject(x, 0-2), and z as cross(x, y).
    Assumptions: 0 and 1 are the carbons, and 2 is a hydrogen bonded to 0."""
    d10 = xyzfi.coords[0] - xyzfi.coords[1]
    d02 = xyzfi.coords[2] - xyzfi.coords[0]
    x_u = JMLMath.unit_vector(d10)
    y = JMLMath.reject_vector(d02, x_u)
    y_u = JMLMath.unit_vector(y)
    assert np.isclose(y_u, JMLMath.unit_vector(JMLMath.reject_vector(d02, d10))).all()
    z_u = np.cross(x_u, y_u)

    carbon_center = 0.5 * (xyzfi.coords[0] + xyzfi.coords[1])
    half_mag_d10 = 0.5 * np.linalg.norm(d10)
    dist = (delta * delta)
    dist -= (half_mag_d10 * half_mag_d10)
    dist = math.sqrt(dist)
    """# TODO: Make an actually elegant calculation
    out_y = carbon_center + ((dist + 1.0) * y_u)"""
    out_z = carbon_center + (dist * z_u)

    center_hy = xyzfi.coords[2] - carbon_center
    d_center_hy = math.sqrt(np.sum(np.square(center_hy)))
    hy_center_probe_angle = JMLMath.vectors_angle(center_hy, y_u)
    probe_hydrogen_center = JMLMath.side_side_angle_triangle(delta, d_center_hy, hy_center_probe_angle)
    if probe_hydrogen_center.ndim == 2:
        assert probe_hydrogen_center[0][0] >= probe_hydrogen_center[1][0]
        # Take the obtuse angle.
        probe_hydrogen_center = probe_hydrogen_center[1]
    out_y = carbon_center + ((probe_hydrogen_center[0] + 0.001) * y_u)

    eprint(f"Moving probe to {out_y}")
    xyzfi.coords[probe_index] = out_y
    xyzfi.write_out("ETHENE-PROBE-2.xyz")

    eprint(f"Moving probe to {out_z}")
    xyzfi.coords[probe_index] = out_z
    xyzfi.write_out("ETHENE-PROBE-3.xyz")


def place_ring(xyzfi: StructXYZ, centroid: np.ndarray, normal: np.ndarray, delta: float = 4.0,
               min_offset: float = 2) -> np.ndarray:
    assert min_offset >= 0
    assert delta > 0
    x = xyzfi.coords
    test_up = place_ring_inner(xyzfi, centroid, normal, delta=delta, min_offset=min_offset)
    test_down = place_ring_inner(xyzfi, centroid, -1 * normal, delta=delta, min_offset=min_offset)
    min_dist_up = 100000
    min_dist_down = 100000

    for i in range(xyzfi.n_atoms):
        dup = dist2(x[i], test_up)
        min_dist_up = min(min_dist_up, dup)
        ddown = dist2(x[i], test_down)
        min_dist_down = min(min_dist_down, ddown)

    if min_dist_down > min_dist_up:
        return test_down
    else:
        return test_up


def place_ring_inner(xyzfi: StructXYZ, centroid: np.ndarray, normal: np.ndarray, delta: float = 4.0,
                     min_offset: float = 2) -> np.ndarray:
    max_r = min_offset
    x = xyzfi.coords

    for i in range(xyzfi.n_atoms):
        # TODO: Just call JMLMath.side_side_angle_triangle.
        # Side-side-angle solution, with known angle theta (angle between normal vector and centroid-atom vector),
        # and distances delta (atom to probe) and dx (atom to centroid). Angle gamma is opposite dx, with alpha
        # opposite the target centroid-probe vector.
        dx = x[i] - centroid
        theta = JMLMath.vectors_angle(dx, normal)
        dist = math.sqrt(np.sum(np.square(dx)))

        sin_g = (dist / delta) * math.sin(theta)
        assert sin_g < 1.000001
        if sin_g >= 1.0:
            # In this case: gamma must be a right angle.
            gamma = 0.5 * math.pi
        elif delta > dist:
            # In this case, the known angle theta must be obtuse, so gamma must be acute.
            gamma = math.asin(sin_g)
        else:
            # In this case, both solutions (acute and obtuse) would satisfy the constraints, but the obtuse solution
            # produces a smaller distance from the centroid.
            gamma = math.pi - math.asin(sin_g)
        alpha = math.pi - (theta + gamma)
        # k: Unknown side length, length of the centroid-probe vector.
        k = delta * (math.sin(alpha) / math.sin(theta))
        max_r = max(k, max_r)

    return centroid + (max_r * normal)


def place_cyclopentane(infile: str, delta=4.0):
    delta = float(delta)
    xyzfi = StructXYZ(infile)
    centroid, normal = JMLMath.best_fit_plane(xyzfi.coords[0:5])
    out_xyz = place_ring(xyzfi, centroid, normal, delta=delta)
    eprint(f"Placing probe at {out_xyz}")
    xyzfi.append_atom(xyzfi.get_default_probetype()[0], out_xyz)
    xyzfi.write_out("CYCLOPENTANE-PROBE.xyz")


def place_halobenzene(infile: str, delta=4.0):
    delta = float(delta)
    xyzfi = StructXYZ(infile)
    assert xyzfi.n_atoms == 12
    for i in range(6):
        assert xyzfi.atom_names[i+1][0] == "C"
    for i in range(5):
        assert xyzfi.atom_names[i+7][0] == "H"
    centroid, normal = JMLMath.best_fit_plane(xyzfi.coords[1:7])
    out_xyz = place_ring(xyzfi, centroid, normal, delta=delta)
    eprint(f"Placing probe at {out_xyz}")
    xyzfi.append_atom(xyzfi.get_default_probetype()[0], out_xyz)
    xyzfi.write_out("HALOBENZENE-PROBE.xyz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('molecule', nargs=1, type=str, help='Name of a molecule with special handling (e.g. AMMONIA)')
    parser.add_argument('infile', nargs=1, type=str, help='Input .xyz file')
    parser.add_argument('additional', nargs='*', help='Any additional arguments')
    args = parser.parse_args()

    molec = args.molecule[0].upper()
    inf = args.infile[0]
    if molec == "AMMONIA":
        ProbePlacers.Ammonia.ammonia(inf, *args.additional)
    elif molec == "WATER":
        ProbePlacers.Water.water(inf, *args.additional)
    elif molec == "BENZENE":
        ProbePlacers.Benzene.benzene(inf, *args.additional)
    elif molec == "ETHENE":
        place_ethene(inf)
    elif molec == "CYCLOPENTANE":
        place_cyclopentane(inf)
    else:
        raise ValueError(f"Could not find any probe placement script with name {args.molecule}!")


if __name__ == "__main__":
    main()
