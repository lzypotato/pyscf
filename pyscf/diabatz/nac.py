from pyscf import gto
from pyscf.diabatz import numerical

class NAC(object):
    '''
    Class for non-adiabatic-coupling (NAC).
    '''
    def __init__(self, mc1, mc2 = None, dR = None, overlapAO = [], nstate1 = 0, nstate2 = 1, method = "numerical"):
        self.mc1 = mc1
        self.nstate1 = nstate1
        self.nstate2 = nstate2
        self.nelecTotal = mc1.mol.nelec[0] + mc1.mol.nelec[1]
        self.ncas = mc1.ncas
        self.nelecas = mc1.nelecas[0] + mc1.nelecas[1]
        self.method = method
        
        if(mc2 == None or dR == None):
            print("ERROR! Analytical methods are NOT supported now.\n You have to input both mc2 and dR")
            exit()
        else:
            self.mc2 = mc2
            self.dR = dR
            
        if(overlapAO != []):
            self.overlapAO = overlapAO
        else:
            mol12 = gto.Mole()
            mol12.basis = mc1.mol.basis
            mol12.atom = mc1.mol.atom + "; " + mc2.mol.atom
            mol12.spin = divmod(mc1.mol.spin + mc2.mol.spin, 2)[1]
            mol12.unit = mc1.mol.unit
            mol12.build()
            self.overlapAO = mol12.intor("int1e_ovlp_sph")
        

    def kernel(self):
        if(self.method == "numerical"):
            return numerical.numerical_NAC(self.mc1, self.mc2, self.nelecTotal, self.ncas, self.nelecas, self.overlapAO, self.dR, self.nstate1, self.nstate2)
        else:
            print("ERROR! Only numerical mthod is supported!")
            exit()
        
