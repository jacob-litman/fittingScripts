#!/usr/bin/env python
# General structure: subtract out unfit ESP (multipoles only) from each fit. Then, use MM probe & match ESP changes

import argparse
from StructureXYZ import StructXYZ

def main():
    parser = argparse.ArgumentParser()
    '''parser.add_argument('-a', dest='atom_index', type=int, required=True, help='Atom to place the probe near')
    parser.add_argument('-n', dest='probe_name', type=str, default='PC', help='Atom name to give the probe')
    parser.add_argument('-t', dest='probe_atype', type=int, default=999, help='Atom type to assign to the probe')
    parser.add_argument('-d', dest='distance', type=float, default=4.0, help='Distance to place the probe at')
    parser.add_argument('-w', dest='hydrogen_weight', type=float, default=0.4, help='Relative weighting for '
                                                                                    'hydrogen distances')
    parser.add_argument('-e', dest='exp', type=int, default=2, help='Exponent for the square of distance in target '
                                                                    'function')
    parser.add_argument('-k', dest='keyfile', type=str, default=None, help='Keyfile to use when saving to PDB')
    parser.add_argument('-x', dest='xyzpdb', type=str, default='xyzpdb', help='Name or full path of Tinker '
                                                                              'xyzpdb')
    parser.add_argument('-o', dest='outfile', type=str, default=None, help='XYZ file to output (if none: '
                                                                           'probe_atom<id>.xyz)')
    parser.add_argument('infile', nargs=1, type=str)'''
    

    args = parser.parse_args()

if __name__ == "__main__":
    main()
