import argparse
from StructureXYZ import StructXYZ
from JMLUtils import eprint
from openbabel import pybel
import numpy as np
from typing import Sequence, Tuple, Dict
import re


custom_smarts = {"A": "[#1,#6]", "B": "[N,O,F,S,Cl,Br]", "D": "[N,O,S]", "d": "[n,n+,o+,s+]", "E": "[#7,O,F,S,Cl,Br]"}


def sub_custom_smarts(smarts: str) -> str:
    ret_str = ""
    for c in smarts:
        ret_str += custom_smarts.get(c, c)
    return ret_str


class PolarType:
    """Defines a polarization type, including a priority, a base SMARTS and position therein, and any SMARTS/positions
    aliased to this type."""
    def __init__(self, toks: Sequence[str]):
        assert len(toks) > 6
        self.id = int(toks[0])
        self.all_ids = [self.id]
        self.smarts_string = sub_custom_smarts(toks[1])
        self.all_smarts_strings = [self.smarts_string]
        self.smarts = pybel.Smarts(self.smarts_string)
        self.all_smarts = [self.smarts]
        self.atom_index = int(toks[2]) - 1
        self.all_indices = [self.atom_index]
        self.initial_polarize = float(toks[4])
        self.polarize = self.initial_polarize
        assert toks[5] == ""
        self.priority = int(toks[6])

    def add_alias_type(self, toks: Sequence[str]):
        self.all_ids.append(int(toks[0]))
        the_smarts = sub_custom_smarts(toks[1])
        self.all_smarts_strings.append(the_smarts)
        self.all_smarts.append(pybel.Smarts(the_smarts))
        self.all_indices.append(int(toks[2]) - 1)
        assert float(toks[4]) == self.initial_polarize
        #assert int(toks[5]) == self.id  # Untrue of indirect aliases.
        assert int(toks[6]) == self.priority

    def match_mol(self, mol: pybel.Molecule) -> Sequence[Tuple[int, int]]:
        """Each element of the returned list is a tuple of (matched atom, matched SMARTS index)."""
        matches = []
        for i, smarts in enumerate(self.all_smarts):
            smart_matches = smarts.findall(mol)
            idx = self.all_indices[i]
            for sm in smart_matches:
                matched_atom = sm[idx]
                matches.append((matched_atom, idx))
        return matches

    def full_str(self) -> str:
        ret_str = self.__str__()
        for i in range(1, len(self.all_smarts), 1):
            ret_str += f"\nSMARTS pattern {i+1}: {self.all_smarts[i]} atom {self.all_indices[i]}"
        return ret_str

    def __str__(self) -> str:
        return f"Polar type {self.id}, polarizability {self.polarize}, priority {self.priority} representing " \
               f"{len(self.all_smarts)} SMARTS patterns.\nDefining SMARTS pattern: {self.smarts} atom {self.atom_index + 1}"


decimal_matcher = re.compile(r'^\d+$')


class PtypeReader:
    def __init__(self, infile: str):
        self.infile = infile
        self.ptypes = dict()
        with open(infile, 'r') as r:
            for line in r:
                toks = line.strip().split("\t")
                eprint(line)
                if decimal_matcher.match(toks[0]):
                    if len(toks[5]) == 0:
                        new_ptype = PolarType(toks)
                        self.ptypes[new_ptype.id] = new_ptype
                        pass
                    else:
                        alias = int(toks[5])
                        try:
                            self.ptypes[alias].add_alias_type(toks)
                        except KeyError as ke:
                            eprint(f"Indirect aliasing (i.e. alias to an alias): {line}")
                            alias_found = False
                            for v in self.ptypes.values():
                                if alias in v.all_ids:
                                    v.add_alias_type(toks)
                                    alias_found = True
                                    break
                            if not alias_found:
                                eprint("Indirect aliasing failed.")
                                raise ke

    def get_ptypes(self) -> Dict[int, PolarType]:
        return self.ptypes

    def __str__(self) -> str:
        ret_str = f"Polar types from {self.infile}"
        for k, v in self.ptypes.values():
            ret_str += f"Atom type {k} pointing to polarizability {v}"
        return ret_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='File to test on.')
    parser.add_argument('ptype_fi', type=str, help='CSV file containing polar type definitions.')

    args = parser.parse_args()
    xyz_s = StructXYZ(args.infile, autogen_mol2=True)
    assert 'PC' not in xyz_s.atom_names
    ptyping = PtypeReader(args.ptype_fi)
    eprint(ptyping)


if __name__ == "__main__":
    main()
