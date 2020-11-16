import sys

import numpy as np
import math

from JMLUtils import dist2, eprint
from StructureXYZ import StructXYZ

TRIANGLE_TOL = 1E-4
Y_DENOM = 1.0 / math.sqrt(3)


def water(infile: str = 'QM_REF.xyz', delta = 4.0):
    delta = float(delta)

    xyzfi = StructXYZ(infile, probe_types=[999])
    assert len(xyzfi.probe_indices) == 0
    assert xyzfi.n_atoms == 3
    assert xyzfi.atom_names[0].startswith("O")

    triangle_center = 0.5 * np.sum(xyzfi.coords[1:3], axis=0)
    place_vec = triangle_center - xyzfi.coords[0]
    mag_pv = math.sqrt(np.dot(place_vec, place_vec))

    bisector_vector = 0.5 * (xyzfi.coords[2] - xyzfi.coords[1])
    # Only used as square, so don't bother w/ square root
    half_bisector = np.dot(bisector_vector, bisector_vector)

    from_bisector = math.sqrt((delta * delta) - half_bisector)
    out_xyz = (place_vec * (from_bisector / mag_pv)) + triangle_center
    eprint(f"Placing probe at {out_xyz}")
    xyzfi.append_atom(xyzfi.get_default_probetype(), out_xyz)
    xyzfi.write_out("WATER_PROBE.xyz")
