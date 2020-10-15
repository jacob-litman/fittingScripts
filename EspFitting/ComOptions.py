import os
from JMLUtils import eprint

DEFAULT_SCRATCH_SUBDIR = 'EspFit'
DEFAULT_SCRATCH_STORAGE = '150GB'
DEFAULT_MEMORY = '12GB'
DEFAULT_NPROC = 4
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 1

class ComOptions:
    def __init__(self, charge: int=DEFAULT_CHARGE, spin: int=DEFAULT_SPIN):
        scrdir = os.environ.get("GAUSS_SCRDIR")
        if scrdir is None:
            eprint("Failed to find environment variable GAUSS_SCRDIR; this must be set before use!")
            self.rwf = None
        else:
            self.rwf = f"{scrdir.rstrip('/')}{os.sep}{DEFAULT_SCRATCH_SUBDIR}{os.sep}"
        self.storage = DEFAULT_SCRATCH_STORAGE
        self.mem = DEFAULT_MEMORY
        self.nproc = DEFAULT_NPROC
        self.chk = 'QM'
        self.method = 'MP2(full)'
        self.basis = 'aug-cc-pVTZ'
        self.scf = 'Save'
        self.density = 'MP2'
        self.guess = 'Huckel'
        self.no_symm = True
        self.charge = True
        self.charge = charge
        self.spin = spin
