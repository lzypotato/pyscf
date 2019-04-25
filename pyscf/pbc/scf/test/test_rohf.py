#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import unittest
import numpy as np
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import krohf

cell = pgto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.basis = '321g'
cell.a = np.eye(3) * 3
cell.mesh = [8] * 3
cell.verbose = 7
cell.output = '/dev/null'
cell.spin = 2
cell.build()

nk = [2, 2, 1]
kpts = cell.make_kpts(nk, wrap_around=True)
kmf = pscf.KROHF(cell, kpts).run()
mf = pscf.ROHF(cell).run()

def tearDownModule():
    global cell, kmf, mf
    cell.stdout.close()
    del cell, kmf, mf

class KnownValues(unittest.TestCase):
    def test_krohf_kernel(self):
        self.assertAlmostEqual(kmf.e_tot, -4.569633030494753, 8)

    def test_rohf_kernel(self):
        self.assertAlmostEqual(mf.e_tot, -3.3633746534777718, 8)

    def test_krhf_vs_rhf(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.ROHF(cell, k, exxdiv='vcut_sph')

        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()

        kmf = pscf.KROHF(cell, [k], exxdiv='vcut_sph')
        kmf.max_cycle = 1
        kmf.diis = None
        e2 = kmf.kernel()
        self.assertAlmostEqual(e1, e2, 9)
        self.assertAlmostEqual(e1, -3.3046228601655607, 9)

    def test_init_guess_by_chkfile(self):
        np.random.seed(1)
        k = np.random.random(3)
        mf = pscf.KROHF(cell, [k], exxdiv='vcut_sph')
        mf.init_guess = 'hcore'
        mf.max_cycle = 1
        mf.diis = None
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -3.4376090968645068, 9)

        mf1 = pscf.ROHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        mf1.diis = None
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -3.4190632006601662, 9)
        self.assertTrue(mf1.mo_coeff[0].dtype == np.double)

    def test_dipole_moment(self):
        dip = mf.dip_moment()
        self.assertAlmostEqual(lib.finger(dip), 1.6424482249196493, 7)

        dip = kmf.dip_moment()
        self.assertAlmostEqual(lib.finger(dip), 0.7361493256233677, 7)

    def test_get_init_guess(self):
        cell1 = cell.copy()
        cell1.dimension = 1
        cell1.build(0, 0)
        mf = pscf.ROHF(cell1)
        dm = mf.get_init_guess(key='minao')
        self.assertAlmostEqual(lib.finger(dm), -0.06586028869608128, 8)

        mf = pscf.KROHF(cell1)
        dm = mf.get_init_guess(key='minao')
        self.assertAlmostEqual(lib.finger(dm), -0.06586028869608128, 8)

    def test_spin_square(self):
        ss = kmf.spin_square()[0]
        self.assertAlmostEqual(ss, 2, 9)

    def test_analyze(self):
        pop, chg = kmf.analyze()
        self.assertAlmostEqual(lib.finger(pop), 1.1120443320325235, 7)
        self.assertAlmostEqual(sum(chg), 0, 7)
        self.assertAlmostEqual(lib.finger(chg), 0.002887875601340767, 7)

if __name__ == '__main__':
    print("Tests for PBC ROHF and PBC KROHF")
    unittest.main()
