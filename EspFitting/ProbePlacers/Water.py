import sys

import numpy as np
import math

from JMLUtils import dist2, eprint
from StructureXYZ import StructXYZ
from typing import Sequence

TRIANGLE_TOL = 1E-4
Y_DENOM = 1.0 / math.sqrt(3)


def water(infile: str = 'QM_REF.xyz', delta=4.0):
    delta = float(delta)

    xyzfi = StructXYZ(infile)
    assert len(xyzfi.probe_indices) == 0
    assert xyzfi.n_atoms == 3
    assert xyzfi.atom_names[0].startswith("O")
    place_triangle(xyzfi, delta)


def place_triangle(xyzfi: StructXYZ, delta: float = 4.0, outname: str = "WATER_PROBE.xyz", center: int = 0, 
                   flank1: int = 1, flank2: int = 2):
    if center >= xyzfi.n_atoms or center < 0:
        raise ValueError(f"Central atom index {center} out-of-bounds 0-{xyzfi.n_atoms}")
    if flank1 >= xyzfi.n_atoms or flank1 < 0:
        raise ValueError(f"Flank1 atom index {flank1} out-of-bounds 0-{xyzfi.n_atoms}")
    if flank2 >= xyzfi.n_atoms or flank2 < 0:
        raise ValueError(f"Flank2 atom index {flank2} out-of-bounds 0-{xyzfi.n_atoms}")
    if center == flank1 or center == flank2 or flank1 == flank2:
        raise ValueError(f"All three atoms must have distinct indices: received {center},{flank1},{flank2}")
    
    triangle_center = xyzfi.coords[center] + xyzfi.coords[flank1] + xyzfi.coords[flank2]
    triangle_center *= 0.5
    place_vec = triangle_center - xyzfi.coords[center]
    mag_pv = math.sqrt(np.dot(place_vec, place_vec))

    bisector_vector = 0.5 * (xyzfi.coords[flank2] - xyzfi.coords[flank1])
    # Only used as square, so don't bother w/ square root
    half_bisector = np.dot(bisector_vector, bisector_vector)

    from_bisector = math.sqrt((delta * delta) - half_bisector)
    out_xyz = (place_vec * (from_bisector / mag_pv)) + triangle_center
    eprint(f"Placing probe at {out_xyz}")
    xyzfi.append_atom(xyzfi.get_default_probetype()[0], out_xyz)
    xyzfi.write_out(outname)
