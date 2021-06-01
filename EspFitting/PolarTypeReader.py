import argparse
from typing import Sequence, Mapping

from openbabel import pybel

import JMLUtils
from JMLUtils import eprint
from StructureXYZ import StructXYZ

custom_smarts = dict()


def sub_custom_smarts(smarts: str) -> str:
    ret_str = ""
    for c in smarts:
        ret_str += custom_smarts.get(c, c)
    return ret_str


class PolarType:
    def __init__(self, def_line: str):
        toks = def_line.strip().split("\t")
        if len(toks) < 6:
            raise ValueError(f"Invalid number of tokens for line {def_line}\nTokens read: {toks}")
        self.id = int(toks[0])
        self.smarts_strings = []
        self.smarts_patts = []
        for subtok in toks[1].split("%"):
            try:
                sm_str = sub_custom_smarts(subtok)
                sm = pybel.Smarts(sm_str)
                self.smarts_strings.append(sm_str)
                self.smarts_patts.append(sm)
            except IOError as ioe:
                eprint(f"Failed to generate SMARTS pattern from tokens {toks}")
                raise ioe

        self.atom_indices = [int(subtok) - 1 for subtok in toks[2].split(",")]
        self.name = toks[3]
        self.initial_polarizability = float(toks[4])
        self.polarizability = self.initial_polarizability
        self.priority = int(toks[5])
        if len(toks) > 6:
            self.enabled = JMLUtils.parse_truth(toks[6])
        else:
            self.enabled = True

    def get_priority(self) -> int:
        return self.priority

    def format_smarts(self, delimiter: str = "%") -> str:
        ret_str = self.smarts_strings[0]
        for i in range(1, len(self.smarts_strings), 1):
            ret_str += f"{delimiter}{self.smarts_strings[i]}"
        return ret_str

    def __str__(self) -> str:
        ret_str = f"Pattern {self.name:<30s}-{self.id:>4d}, priority {self.priority:>2d}, initial polarizability " \
               f"{self.initial_polarizability:7.4f}, enabled {self.enabled}, SMARTS {self.format_smarts(delimiter=',')}"
        return ret_str

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __gt__(self, other):
        return self.id > other.id

    # TODO: More robust hash, particularly ensuring id gets replaced by an immutable value.
    def __hash__(self):
        return hash(self.id)


class PtypeReader:
    def __init__(self, infile: str):
        self.infile = infile
        self.ptypes = []
        self.id_ptypes = dict()
        with open(infile, 'r') as r:
            for line in r:
                line = line.strip()
                if line == "" or line.startswith("ID") or line.startswith("#"):
                    continue
                new_ptype = PolarType(line)
                self.ptypes.append(new_ptype)
                self.id_ptypes[new_ptype.id] = new_ptype

    def all_match_mol(self, mol: StructXYZ, verbose: bool = False) -> Sequence[Sequence[PolarType]]:
        obm = mol.ob_rep
        if obm is None:
            raise ValueError(f"Could not find OpenBabel representation for {mol}!")
        matches = [[] for _ in range(mol.n_atoms)]

        for pt in self.ptypes:
            smart_matches = []
            for sm in pt.smarts_patts:
                smart_matches.extend(sm.findall(obm))
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
            if n_match == 0:
                raise ValueError(f"No matches for molecule {mol}")
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


def main_inner(xyz_s: StructXYZ, verbose: bool, ptype_fi: str = None, ptyping: PtypeReader = None,
               fail_on_multimatch: bool = False) -> (Sequence[PolarType], Mapping[int, int], PtypeReader):
    """Return values: ptypes is a list of each atom's assigned polar type, ptype_map is the mapping of atom type to
    polar type ID, and ptyping is either the ptyping parameter (if ptyping is not None), or the auto-generated
    PtypeReader (if ptyping is None)."""
    assert 'PC' not in xyz_s.atom_names
    if ptyping is None:
        assert ptype_fi is not None
        ptyping = PtypeReader(ptype_fi)

    ptypes = ptyping.match_mol(xyz_s, verbose)
    ptype_map = dict()
    for i, pt in enumerate(ptypes):
        atype = xyz_s.assigned_atypes[i]
        if atype in ptype_map:
            if ptype_map[atype] != pt.id:
                message = f"Attempted to assign polar type {pt.id} to type {atype}, already assigned to " \
                          f"{ptype_map[atype]}!"
                if fail_on_multimatch:
                    raise ValueError(message)
                else:
                    eprint(f"WARNING: {message}")
                    prior_ptype = ptyping.ptypes[ptype_map[atype]]
                    if pt.priority > prior_ptype.priority:
                        eprint(f"Updating mapping for atom type {atype} from {ptype_map[atype]} to {pt.id}")
                        ptype_map[atype] = pt.id
            else:
                ptype_map[xyz_s.assigned_atypes[i]] = pt.id
        else:
            ptype_map[xyz_s.assigned_atypes[i]] = pt.id
    if verbose:
        eprint(f"Mapping: {ptype_map}")
    return ptypes, ptype_map, ptyping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='File to test on.')
    parser.add_argument('ptype_fi', type=str, help='CSV file containing polar type definitions.')
    parser.add_argument('--verbose', action='store_true', help='Print out all matches, not just the highest-priority '
                                                               'matches')
    args = parser.parse_args()
    xyz_s = StructXYZ(args.infile)
    ptypes, mapping, ptyping = main_inner(xyz_s, args.ptype_fi, args.verbose)

    eprint("\n")
    for i, at in enumerate(xyz_s.assigned_atypes):
        eprint(f"Atom:                  {xyz_s.atom_names[i]}-{i+1:d}")
        eprint(f"  Type:                {at}")
        eprint(f"  Polar type in array: {ptypes[i]}")
        if args.verbose:
            eprint(f"  Mapped to:           {mapping[at]}")
        if mapping[at] != ptypes[i].id:
            raise ValueError(f"Atom {i+1} had a bad mapping!")
        eprint("")


if __name__ == "__main__":
    main()
