import argparse
from typing import Sequence

from openbabel import pybel

from JMLUtils import eprint
from StructureXYZ import StructXYZ

#custom_smarts = {"B": "[N,O,F,S,Cl,Br]", "D": "[N,O,S]", "d": "[n,n+,o+,s+]", "E": "[#7,O,F,S,Cl,Br]"}
custom_smarts = dict()


def sub_custom_smarts(smarts: str) -> str:
    ret_str = ""
    for c in smarts:
        ret_str += custom_smarts.get(c, c)
    return ret_str


class PolarType:
    def __init__(self, def_line: str):
        toks = def_line.strip().split("\t")
        assert len(toks) == 6
        self.id = int(toks[0])
        self.smarts_string = sub_custom_smarts(toks[1])
        try:
            self.smarts = pybel.Smarts(self.smarts_string)
        except IOError as ioe:
            eprint(f"Failed to generate SMARTS pattern from tokens {toks}")
            raise ioe
        #self.atom_index = int(toks[2]) - 1
        self.atom_indices = [int(subtok) - 1 for subtok in toks[2].split(",")]
        self.name = toks[3]
        self.initial_polarizability = float(toks[4])
        self.polarizability = self.initial_polarizability
        self.priority = int(toks[5])

    def get_priority(self) -> int:
        return self.priority

    def __str__(self) -> str:
        return f"Pattern {self.name:<30s}-{self.id:>4d}, priority {self.priority:>2d}, initial polarizability " \
               f"{self.initial_polarizability:7.4f}, SMARTS {self.smarts_string}"


class PtypeReader:
    def __init__(self, infile: str):
        self.infile = infile
        self.ptypes = []
        with open(infile, 'r') as r:
            for line in r:
                if not line.startswith("ID"):
                    self.ptypes.append(PolarType(line))

    def all_match_mol(self, mol: StructXYZ, verbose: bool = False) -> Sequence[Sequence[PolarType]]:
        obm = mol.ob_rep
        assert obm is not None
        matches = [[] for _ in range(mol.n_atoms)]

        for pt in self.ptypes:
            smart_matches = pt.smarts.findall(obm)
            for match in smart_matches:
                # Assign this match to the first atom index listed, and to subsequent indices IFF they are of the same element.
                atom = match[pt.atom_indices[0]] - 1
                matches[atom].append(pt)
                el = mol.get_atomic_no(atom)
                for i in range(1, len(pt.atom_indices), 1):
                    ai = pt.atom_indices[i]
                    atom = match[ai] - 1
                    if el == mol.get_atomic_no(atom):
                        matches[atom].append(pt)

        for i, m in enumerate(matches):
            m.sort(key=PolarType.get_priority, reverse=True)
            if (verbose):
                eprint(f"{len(m)} matches for atom {mol.atom_names[i]}-{i}:")
                for mj in m:
                    eprint(f"    Matched {mj}")
        return matches

    def match_mol(self, mol: StructXYZ, verbose: bool = False) -> Sequence[PolarType]:
        all_matches = self.all_match_mol(mol, verbose)
        nonredundant = []
        for i, m_set in enumerate(all_matches):
            n_match = len(m_set)
            assert n_match > 0
            priority = m_set[0].priority
            id = m_set[0].id
            for i in range(1, n_match, 1):
                if m_set[i].priority < priority:
                    break
                elif m_set[i].id != id:
                    raise ValueError(f"Multiple equal-priority, non-identical matches for atom {mol.atom_names[i]}-{i} "
                                     f"in {mol}:\n{m_set}")
            nonredundant.append(m_set[0])
        return nonredundant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='File to test on.')
    parser.add_argument('ptype_fi', type=str, help='CSV file containing polar type definitions.')
    parser.add_argument('--verbose', action='store_true', help='Print out all matches, not just the highest-priority '
                                                               'matches')

    args = parser.parse_args()
    xyz_s = StructXYZ(args.infile)
    assert 'PC' not in xyz_s.atom_names
    ptyping = PtypeReader(args.ptype_fi)
    ptypes = ptyping.match_mol(xyz_s, args.verbose)

    eprint("\n")
    for ptype in ptypes:
        eprint(ptype)


if __name__ == "__main__":
    main()
