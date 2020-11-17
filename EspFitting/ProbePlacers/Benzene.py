import numpy as np
from StructureXYZ import StructXYZ
from JMLUtils import eprint
import math

def benzene(infile: str = 'QM_REF.xyz', delta=4.0):
    delta = float(delta)
    xyzfi = StructXYZ(infile)
    assert xyzfi.n_atoms >= 12
    for i in range(6):
        assert xyzfi.atom_names[i].startswith("C")
        assert xyzfi.atom_names[i+6].startswith("H")
    ring_center = np.mean(xyzfi.coords[0:6], axis=0)
    eprint(f"Ring center is at {ring_center}")

    vec1 = xyzfi.coords[2] - xyzfi.coords[0]
    vec2 = xyzfi.coords[4] - xyzfi.coords[0]
    vec3 = np.cross(vec1, vec2)
    mag = math.sqrt(np.dot(vec3, vec3))
    vec3 *= (delta / mag)

    eprint(f"Displacement vector: {vec3}")
    probe_loc = vec3 + ring_center
    eprint(f"Probe placement: {probe_loc[0]:.8f} {probe_loc[1]:.8f} {probe_loc[2]:.8f}")
    """assert len(xyzfi.probe_indices) == 0
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
    xyzfi.append_atom(xyzfi.get_default_probetype()[0], out_xyz)
    xyzfi.write_out("AMMONIA_PROBE.xyz")"""