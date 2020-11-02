import re
import os
from JMLUtils import eprint

DEFAULT_OPTIONS = {'numproc': '4', 'maxmem': '12GB', 'maxdisk': '100GB', 'espmethod': 'PBE0',
                   'espbasisset': 'aug-cc-pVTZ', 'program': 'psi4'}
DEFAULT_FILENAMES = frozenset(['poltype.ini', 'espfit.ini'])


class OptParser:
    def rectify_oddities(self):
        """Should be called upon modification to deal with gotchas like Gaussian using PBE1PBE to represent the PBE0
        hybrid DFT functional. Also standardizes some capitalization (e.g. psi4 to PSI4)"""
        self.options['program'] = self.options['program'].upper()
        if self.options['program'].startswith('GAUSS'):
            if self.options['espmethod'].upper() == 'PBE0':
                eprint("Gaussian is in use with PBE0 method specified: altering to Gaussian-recognized PBE1PBE keyword.")
                self.options['espmethod'] = 'PBE1PBE'
        elif self.options['program'] == 'PSI4':
            if self.options['espmethod'].upper() == 'PBE1PBE':
                eprint("Psi4 is in use: altering Gaussian keyword PBE1PBE to literature naming of PBE0.")
                self.options['espmethod'] = 'PBE0'

    def __init__(self, fname: str = None):
        self.options = DEFAULT_OPTIONS.copy()
        if fname is None:
            for fn in DEFAULT_FILENAMES:
                if os.path.exists(fn) and os.path.isfile(fn):
                    fname = fn
                    break
        if fname is not None:
            eprint(f'Reading options from {fname}')
            with open(fname, 'r') as r:
                for line in r:
                    toks = re.sub(r'\s*#.*', '', line).strip().split('=')
                    if len(toks) == 2:
                        self.options[toks[0]] = toks[1]
        else:
            eprint('No options file specified or found; default properties will be used!')
        self.rectify_oddities()
