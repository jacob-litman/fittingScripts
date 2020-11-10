import numpy as np
import math

from JMLUtils import eprint
from StructureXYZ import StructXYZ

TRIANGLE_TOL = 1E-4
Y_DENOM = 1.0 / math.sqrt(3)


def ammonia(infile: str = 'QM_REF.xyz', delta = 4.0):
    delta = float(delta)

    xyzfi = StructXYZ(infile, probe_types=[999])
    assert len(xyzfi.probe_indices) == 0
    assert xyzfi.n_atoms == 4
    assert xyzfi.atom_names[0].startswith("N")

    triangle_center = 1/3 * np.sum(xyzfi.coords[1:4], axis=0)
    place_vec = triangle_center - xyzfi.coords[0]
    mag_pv = math.sqrt(np.dot(place_vec, place_vec))

    center_to_vertex = xyzfi.coords[1] - triangle_center
    # mag_c2v will only be used in squared form.
    mag_c2v = np.dot(center_to_vertex, center_to_vertex)
    mag_place = math.sqrt((delta * delta) - mag_c2v)

    out_xyz = (place_vec * (mag_place / mag_pv)) + triangle_center
    eprint(f"Placing probe at {out_xyz}")
    xyzfi.append_atom("PC", out_xyz, 999)
    xyzfi.write_out("AMMONIA_PROBE.xyz")
