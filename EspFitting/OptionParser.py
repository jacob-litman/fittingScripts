import re
import os
from JMLUtils import eprint, parse_qm_program, QMProgram

DEFAULT_OPTIONS = {'numproc': '4', 'maxmem': '12GB', 'maxdisk': '100GB', 'espmethod': 'PBE0',
                   'espbasisset': 'aug-cc-pVTZ', 'program': 'psi4'}
DEFAULT_FILENAMES = frozenset(['poltype.ini', 'espfit.ini'])
# Below largely scraped from Gaussian documentation: http://gaussian.com/dft/
KNOWN_DFT_TYPES = ["B3LYP", "B3P86", "B3PW91", "O3LYP", "AFPD", "AFP", "wB97XD", "LC-wHPBE", "CAM-B3LYP", "wB97",
                   "wB97X", "MN15", "MN11", "SOGGA11X", "NI2SX", "MN12SX", "PW6B95", "PW6B95D3", "M08HX", "M06",
                   "M06HF", "M062X", "M05", "M052X", "PBE0", "PBE1PBE", "HSEH1PBE", "OHSE2PBE", "OHSE1PBE", "PBEh1PBE",
                   "B1B95", "B1LYP", "mPW1PW91", "mPW1LYP", "mPW1PBE", "mPW3PBE", "B98", "B971", "B972", "TPSSh",
                   "tHCTHhyb", "BMK", "HISSbPBE", "X3LYP", "BHandH", "BHandHLYP"]
scf_types = KNOWN_DFT_TYPES.copy()
scf_types.append('HF')
KNOWN_SCF_TYPES = frozenset(scf_types)


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
        self.file = fname
        self.rectify_oddities()

    def is_psi4(self) -> bool:
        return self.get_program() == QMProgram.PSI4

    def get_program(self) -> QMProgram:
        return parse_qm_program(self.options['program'])

    def gauss_potential_type(self) -> str:
        espm = re.sub(r'^R', '', self.options['espmethod'])
        if espm.startswith("MP2"):
            return "MP2"
        elif espm.startswith("CI"):
            return "CI"
        elif espm.startswith("CC") or espm.startswith("QCI"):
            return "CC"
        elif espm in KNOWN_SCF_TYPES:
            return "SCF"
        else:
            eprint(f"ESP method {self.options['espmethod']} is not recognized; defaulting to potential=SCF")
            return "SCF"
