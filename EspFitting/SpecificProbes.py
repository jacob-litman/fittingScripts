import argparse
import ProbePlacers.Ammonia
import ProbePlacers.Water
import sys
from JMLUtils import eprint


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
    else:
        raise ValueError(f"Could not find any probe placement script with name {args.molecule}!")


if __name__ == "__main__":
    main()
