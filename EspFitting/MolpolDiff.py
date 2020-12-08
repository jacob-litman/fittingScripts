#!/usr/bin/env python
import argparse
import numpy as np
import re
from JMLUtils import parse_gauss_float, gauss_polar_convert
from enum import Enum, auto

mname_patt = re.compile(r'^(\d{3})_.+_.+$')


class MolpolLog(Enum):
    GAUSSIAN = auto()
    TINKER = auto()
    CSV = auto()


def get_mp_logtype(fname: str) -> MolpolLog:
    with open(fname, 'r') as r:
        for line in r:
            if "Gaussian" in line:
                return MolpolLog.GAUSSIAN
            elif line == "Index,MolName,Isotropic,Anisotropic,xx,yx,yy,zx,zy,zz":
                return MolpolLog.CSV
            elif "Tinker  ---  Software Tools for Molecular Design" in line:
                return MolpolLog.TINKER
    raise ValueError(f"Could not determine the type (originating program) of {fname}")


looks_like_patt = re.compile(r'^([0-9]{3})_[^.]+\.log$')
polar_el_patt = re.compile(r'^ +([a-z]+) +(-?\d+\.\d+D[+\-]\d{2}) +(-?\d+\.\d+D[+\-]\d{2}) +(-?\d+\.\d+D[+\-]\d{2}) *$')
p_el_map = {"iso": 0, "xx": 1, "yx": 2, "yy": 3, "zx": 4, "zy": 5, "zz": 6}


def parse_gauss_mpol(fname: str) -> np.ndarray:
    ret_arr = np.full(7, np.nan, dtype=np.float64)
    with open(fname, 'r') as r:
        found_polarize = False
        for line in r:
            if found_polarize:
                m = polar_el_patt.match(line)
                if m:
                    to_index = p_el_map.get(m.group(1), -1)
                    if to_index >= 0:
                        out_val = parse_gauss_float(m.group(2)) * gauss_polar_convert
                        ret_arr[to_index] = out_val
                elif line.strip() == "----------------------------------------------------------------------":
                    found_polarize = False
            elif line.strip() == "Dipole polarizability, Alpha (input orientation).":
                found_polarize = True
    assert True not in np.isnan(ret_arr)
    return ret_arr


def parse_tinker_mpol(fname: str) -> np.ndarray:
    ret_arr = np.full(7, np.nan, dtype=np.float64)
    with open(fname, 'r') as r:
        line = r.readline()
        while line != "":
            line = line.strip()
            if line == "Molecular Polarizability Tensor :":
                r.readline()
                # Isotropic,xx,yx,yy,zx,zy,zz
                toks = r.readline().strip().split()
                ret_arr[1] = float(toks[0])
                ret_arr[2] = float(toks[1])
                ret_arr[4] = float(toks[2])
                toks = r.readline().strip().split()
                assert ret_arr[2] == float(toks[0])
                ret_arr[3] = float(toks[1])
                ret_arr[5] = float(toks[2])
                toks = r.readline().strip().split()
                assert ret_arr[4] == float(toks[0])
                assert ret_arr[5] == float(toks[1])
                ret_arr[6] = float(toks[2])
            elif line.startswith("Interactive Molecular Polarizability :"):
                ret_arr[0] = float(line.split()[4])
            line = r.readline()
    assert True not in np.isnan(ret_arr)
    return ret_arr


def parse_csv_mpol(fname: str, molnum: int = None) -> np.ndarray:
    raise NotImplementedError("Not yet implemented!")


def parse_mp_log(fname: str, mp_logtype: MolpolLog = None) -> np.ndarray:
    """Returns array of isotropic polarizability, xx, xy, yy, xz, yz, zz polarizabilities"""
    if mp_logtype is None:
        mp_logtype = get_mp_logtype(fname)
    if mp_logtype == MolpolLog.TINKER:
        return parse_tinker_mpol(fname)
    elif mp_logtype == MolpolLog.GAUSSIAN:
        return parse_gauss_mpol(fname)
    elif mp_logtype == MolpolLog.CSV:
        return parse_csv_mpol(fname)
    else:
        raise ValueError(f"Could not recognize molecular polarizability type {mp_logtype}")


def diff_arr(fit_mp_fi: str, fit_mp_type: MolpolLog, ref_mp_fi: str,
             ref_mp_type: MolpolLog) -> (np.ndarray, np.ndarray, np.ndarray):
    fit = parse_mp_log(fit_mp_fi, fit_mp_type)
    ref = parse_mp_log(ref_mp_fi, ref_mp_type)
    diff = fit - ref
    return fit, ref, diff


def format_arr(the_arr: np.ndarray, title: str) -> str:
    return f"{title},{the_arr[0]:.9f},{the_arr[1]:.9f},{the_arr[2]:.9f},{the_arr[3]:.9f},{the_arr[4]:.9f}," \
           f"{the_arr[5]:.9f}"


def main_inner(fit_mp_fi: str, ref_mp_fi: str, fit_mp_type: MolpolLog = None, ref_mp_type: MolpolLog = None):
    fit, ref, diff = diff_arr(fit_mp_fi, fit_mp_type, ref_mp_fi, ref_mp_type)
    print("Polarizability,Isotropic,xx,xy,yy,xz,yz,zz")
    print(format_arr(fit, "Fit"))
    print(format_arr(ref, "Reference"))
    print(format_arr(diff, "Difference"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fit_mpol', type=str, help='Logfile with molecular polarizabilities.')
    parser.add_argument('ref_mpol', type=str, help='Logfile with reference molecular polarizabilities.')
    args = parser.parse_args()
    main_inner(args.fit_mpol, args.ref_mpol)


if __name__ == "__main__":
    main()
