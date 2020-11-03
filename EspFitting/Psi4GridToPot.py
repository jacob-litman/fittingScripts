from JMLUtils import version_file, hartree
import numpy as np


def psi4_grid2pot(outfile: str, method: str = None, comments: str = None, grid: str = 'grid.dat',
                  esp: str = 'grid_esp.dat', do_version: bool = False) -> str:
    if do_version:
        outfile = version_file(outfile)
    gridarr = np.genfromtxt(grid, dtype=np.float64)
    esparr = np.genfromtxt(esp, dtype=np.float64) * hartree
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
