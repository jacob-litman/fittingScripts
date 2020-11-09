from JMLUtils import version_file, hartree
import numpy as np
import argparse
import scipy.stats
import math


def psi4_grid2pot(outfile: str, method: str = None, comments: str = None, grid: str = 'grid.dat',
                  esp: str = 'grid_esp.dat', do_version: bool = False, verbose: bool = True) -> str:
    if do_version:
        outfile = version_file(outfile)
    gridarr = np.genfromtxt(grid, dtype=np.float64)
    esparr = np.genfromtxt(esp, dtype=np.float64) * hartree
    if verbose:
        desc = scipy.stats.describe(esparr)
        sd = math.sqrt(desc[3])
        print(f"Mean electrostatic potential: {desc[2]:.5g} with standard deviation {sd:.5g}")
    # TODO: Conversion factor.
    n_points = esparr.shape[0]
    assert n_points == gridarr.shape[0]
    if method is None:
        method = 'Unknown QM method'
    if comments is None:
        comments = "Psi4 ESP (electrostatic potential)"

    with open(outfile, 'w') as w:
        w.write(f'{n_points:>8d}  {method} {comments}\n')
        for i in range(n_points):
            w.write(f"{i + 1:>8d}   ")
            for j in range(3):
                w.write(f" {gridarr[i][j]:>11.6f}")
            w.write(f" {esparr[i]:13.4f}\n")
    return outfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='output_file', type=str, default='esp.pot', help='File to write to')
    parser.add_argument('-c', dest='comments', type=str, nargs='*', action='append', help='String to write to header '
                                                                                          'line')
    parser.add_argument('-g', dest='grid_fi', type=str, default='grid.dat', help='Grid file to read from')
    parser.add_argument('-e', dest='esp_fi', type=str, default='grid_esp.dat', help='ESP file to read from')
    parser.add_argument('--version', dest='version', action='store_true',
                        help='Version output (rather than clobbering output file)')
    parser.add_argument('-q', dest='quiet', action='store_false', help='Do not print logging information')
    args = parser.parse_args()

    if args.comments is not None and len(args.comments) > 0:
        comments = ""
        for c in args.comments:
            comments += c.replace('\n', ' ')
    else:
        comments = None

    psi4_grid2pot(outfile=args.output_file, comments=comments, grid=args.grid_fi, esp=args.esp_fi,
                  do_version=args.version, verbose=(not args.quiet))


if __name__ == "__main__":
    main()
