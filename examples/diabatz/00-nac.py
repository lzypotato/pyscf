import pyscf, numpy, os
from pyscf import gto, scf, mcscf, diabatz

ncas=          8
nelecas=          7 
nstate1=           0
nstate2=           1
dR=       0.0099999997764826

mol1 = gto.Mole()                                                
mol1.build(                                                      
output = None,                                                   
symmetry = False,                                                
unit = "Bohr",                                                   
atom =                                                           
'''                                                              
Li             0.000000            4.103952            2.873618  
F             0.000000            0.000000            0.000000   
H             0.000000            4.095760           -2.867882   
;                                                                
'''                                                              
,                                                                
spin = 1,                                                        
basis = 'cc-pvtz'                                                
                                                                 
)                                                                
mol2 = gto.Mole()                                                
mol2.build(                                                      
output = None,                                                   
symmetry = False,                                                
unit = "Bohr",                                                   
atom =                                                           
'''                                                              
Li             0.000000            4.087569            2.862146  
F             0.000000            0.000000            0.000000   
H             0.000000            4.095760           -2.867882   
;                                                                
'''                                                              
,                                                                
spin = 1,                                                        
basis = 'cc-pvtz'                                                
                                                                 
)                                                                
mf1 = scf.RHF(mol1)
mf1.conv_tol = 1e-9
mf1.scf()
mc1 = mcscf.CASSCF(mf1, ncas, nelecas).state_average_(numpy.ones((2))/2.0)                         
mc1.kernel()

mf2 = scf.RHF(mol2)
mf2.conv_tol = 1e-9
mf2.scf()
mc2 = mcscf.CASSCF(mf2, ncas, nelecas).state_average_(numpy.ones((2))/2.0)                         
mc2.kernel()

mynac = diabatz.nac.NAC(mc1, mc2, dR = dR)
nac, overlap = mynac.kernel()
print (nac)
print (overlap[0][0],overlap[0][1])
print (overlap[1][0],overlap[1][1])
mo_range = (mc1.mol.nelec[0]+mc1.mol.nelec[1]-nelecas)//2 + ncas
with open('mc1.txt', 'w') as file_object1:
  for ii in range(mo_range):
        for kk in mc1.mo_coeff[..., ii]:
              file_object1.write(str(kk))
              file_object1.write("  ")
        file_object1.write("\n")
  for ii in [nstate1,nstate2]:
        for jj in range(len(mc1.ci[0])):
              for kk in mc1.ci[ii][jj]:
                    file_object1.write(str(kk))
                    file_object1.write("  ")
        file_object1.write("\n")

with open('mc2.txt', 'w') as file_object2:
  for ii in range(mo_range):
        for kk in mc2.mo_coeff[..., ii]:
              file_object2.write(str(kk))
              file_object2.write("  ")
        file_object2.write("\n")
  for ii in [nstate1,nstate2]:
        for jj in range(len(mc2.ci[0])):
              for kk in mc2.ci[ii][jj]:
                    file_object2.write(str(kk))
                    file_object2.write("  ")
        file_object2.write("\n")
