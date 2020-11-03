import os
from JMLUtils import eprint
from OptionParser import OptParser

DEFAULT_SCRATCH_SUBDIR = 'EspFit'
DEFAULT_SCRATCH_STORAGE = '150GB'
DEFAULT_MEMORY = '12GB'
DEFAULT_NPROC = 4
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 1
DEFAULT_PROGRAM = 'PSI4'


class ComOptions:
    def __init__(self, charge: int=DEFAULT_CHARGE, spin: int=DEFAULT_SPIN, opts: OptParser = None):
        self.storage = DEFAULT_SCRATCH_STORAGE
        self.mem = DEFAULT_MEMORY
        self.nproc = DEFAULT_NPROC
        self.chk = 'QM'
        self.method = 'MP2(full)'
        self.basis = 'aug-cc-pVTZ'
        self.scf = 'Save'
        self.density = 'Current'
        self.guess = 'Huckel'
        self.no_symm = True
        self.charge = True
        self.charge = charge
        self.spin = spin
        self.program = DEFAULT_PROGRAM
        self.do_polar = False
        self.write_esp = True
        if opts is not None:
            opd = opts.options
            self.storage = opd['maxdisk']
            self.mem = opd['maxmem']
            self.nproc = opd['numproc']
            self.method = opd['espmethod']
            self.basis = opd['espbasisset']
            self.program = opd['program']

        if self.program.startswith('GAUSS'):
            scrdir = os.environ.get("GAUSS_SCRDIR")
            if scrdir is None:
                eprint("WARNING: Failed to find environment variable GAUSS_SCRDIR: falling back to PSI4_SCRATCH")
                scrdir = os.environ.get("PSI_SCRATCH")
            if scrdir is None:
                eprint("ERROR: Failed to find environment variables GAUSS_SCRDIR and PSI4_SCRATCH; one must be set before use!")
                self.rwf = None
                # TODO: Raise appropriate Error.
            else:
                self.rwf = f"{scrdir.rstrip('/')}{os.sep}{DEFAULT_SCRATCH_SUBDIR}{os.sep}"
        elif self.program == 'PSI4':
            scrdir = os.environ.get("PSI_SCRATCH")
            if scrdir is None:
                eprint("WARNING: Failed to find environment variable PSI4_SCRATCH: falling back to GAUSS_SCRDIR")
                scrdir = os.environ.get("GAUSS_SCRDIR")
            if scrdir is None:
                eprint("ERROR: Failed to find environment variables PSI4_SCRATCH and GAUSS_SCRDIR; one must be set before use!")
                self.rwf = None
                # TODO: Raise appropriate Error.
            else:
                self.rwf = f"{scrdir.rstrip('/')}{os.sep}{DEFAULT_SCRATCH_SUBDIR}{os.sep}"
