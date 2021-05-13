import argparse
import numpy as np
import os.path
from os.path import join

import EspAnalysis
import EspFitPolar
from JMLUtils import eprint


def isfile_safe(path: str = None):
    if path is None:
        return False
    return os.path.isfile(path)


def del_esp(esp1_fi: str, esp2_fi: str, power: int = 2) -> float:
    #qme = np.genfromtxt(join(to_sdir, 'qm_polarization.pot'), usecols=pot_cols, skip_header=1, dtype=np.float64)
    esp1 = np.genfromtxt(esp1_fi, usecols=[4], skip_header=1, dtype=np.float64)
    esp2 = np.genfromtxt(esp2_fi, usecols=[4], skip_header=1, dtype=np.float64)
    d_esp = esp1 - esp2
    if power % 2 == 0:
        energy = np.average(np.power(d_esp, power))
    else:
        energy = np.average(np.abs(np.power(d_esp, power)))
    return energy

def del_tensor(mpol1_fi: str, mpol2_fi: str, power: int = 2,
               mpol_component_weights: np.ndarray = EspFitPolar.DEFAULT_MPOL_COMPONENT_WEIGHTS) -> float:
    mpol1 = EspAnalysis.qm_polarization_tensor(mpol1_fi)
    mpol2 = EspAnalysis.qm_polarization_tensor(mpol2_fi)
    d_mpol = mpol1 - mpol2
    d_mpol = np.power(d_mpol, power)
    if power % 2 == 1:
        d_mpol = np.abs(d_mpol)
    d_mpol *= mpol_component_weights
    return np.sum(d_mpol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--molpol_wt', dest='mpol_wt', type=float, default=1.0,
                        help='Relative weighting of molecular polarizabilitity error')
    parser.add_argument('--ew', '--esp_wt', dest='esp_wt', type=float, default=1.0,
                        help='Relative weighting of electrostatic potential error')
    parser.add_argument('--e1', '--esp1', dest='esp1_fi', type=str, default=None, help='Reference potential file')
    parser.add_argument('--e2', '--esp2', dest='esp2_fi', type=str, default=None,
                        help='Target potential file (subtracted from esp1)')
    parser.add_argument('--m1', '--molpol1', dest='mpol1_fi', type=str, default=None,
                        help='Reference molecular polarizabilities file')
    parser.add_argument('--m2', '--molpol2', dest='mpol2_fi', type=str, default=None,
                        help='Target molecular polarizabilities file (subtracted from molpol1)')
    args = parser.parse_args()

    if isfile_safe(args.esp1_fi) and isfile_safe(args.esp2_fi):
        do_esp = True
        assert args.esp_wt >= 0.0
    else:
        do_esp = False

    if isfile_safe(args.mpol1_fi) and isfile_safe(args.mpol2_fi):
        do_mpol = True
        assert args.mpol_wt >= 0.0
    else:
        do_mpol = False

    if not do_esp and not do_mpol:
        eprint(f"No differences to calculate: valid file pairs not provided!")

    target = 0
    if do_esp:
        de = del_esp(args.esp1_fi, args.esp2_fi)
        reweighted = EspFitPolar.DEFAULT_ESP_WT * de * args.esp_wt
        eprint(f"Average ESP difference: {de:.6f}, reweighted to {reweighted:.6f}")
        target += reweighted

    if do_mpol:
        dm = del_tensor(args.mpol1_fi, args.mpol2_fi)
        reweighted = EspFitPolar.DEFAULT_MPOL_WT * dm * args.mpol_wt
        eprint(f"Molecular polarizability difference: {dm:.6f}, reweighted to {reweighted:.6f}")
        target += reweighted

    eprint(f"Total target function value: {target:.6f}")


if __name__ == "__main__":
    main()