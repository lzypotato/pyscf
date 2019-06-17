import numpy, scipy, itertools
from scipy.special import comb

def numerical_NAC(mc1, mc2, nelec, ncas, nelecas, overlapAllAO, dR, nstate11 = 0, nstate12 = 1, nstate21 = 0, nstate22 = 1):
    '''
    Numerical non-adiabatic-coupling calculator for multi-electrons-wavefunction.
    It is based on equ(5) in J. Phys. Chem. Lett. 2015, 6, 4200-4203.

    overlapAllAO:   Overlap between all atomic orbitals from mol12, which can be divided into 
                    overlap between mol1 & mol1, mol1 & mol2, mol2 & mol1 and mol2 & mol2. 
                    
                    Only the second one will be used for calculate overlapMO.
    
    overlapAO:      Second part of overlapAllAO.
    
    overlapMO:      Overlap between all molecular orbitals between mol1 and mol2.

    overlapCAS:     Overlap between mcscf wavefunction. 
    '''
    overlapAO = overlapAllAO[:len(overlapAllAO)//2, len(overlapAllAO)//2:]
    nelecore = nelec - nelecas
    overlapMO = numpy.zeros((nelecore//2+ncas, nelecore//2+ncas))
    
    # Convert overlapAO to overlapMO
    for ii in range(nelecore//2+ncas):
        for jj in range(nelecore//2+ncas):
            moTmp1 = mc1.mo_coeff[..., ii]
            moTmp2 = mc2.mo_coeff[..., jj]
            overlapMO[ii][jj] = numpy.einsum("i,ij,j->", moTmp1, overlapAO, moTmp2)
    
    # Convert ci_coeff matrix to one-dimensional array
    ciRow = mc1.ci[nstate1].shape[0]
    ciColumn = mc1.ci[nstate1].shape[1]

    ci1 = numpy.zeros((2,ciRow*ciColumn))
    ci2 = numpy.zeros((2,ciRow*ciColumn))
    for ii in range(len(ci1[0])):
        ci1[0][ii] = mc1.ci[nstate11][ii//ciColumn][ii-ii//ciColumn*ciColumn]
        ci2[0][ii] = mc2.ci[nstate21][ii//ciColumn][ii-ii//ciColumn*ciColumn]
        ci1[1][ii] = mc1.ci[nstate12][ii//ciColumn][ii-ii//ciColumn*ciColumn]
        ci2[1][ii] = mc2.ci[nstate22][ii//ciColumn][ii-ii//ciColumn*ciColumn]
    
    # Calculate overlapCAS and NAC using equ(5) in the literature
    overlapCAS = get_ovlp_nonothorgonal(ci1, ci2, ncas, nelecas, nelec, overlapMO)
    NAC = (overlapCAS[1][0]-overlapCAS[0][1])/4.0/dR

    return NAC, overlapCAS


def get_ovlp_nonothorgonal(ci1, ci2, ncas, nelecas, nelec, overlapMO):
    '''
    Calculate overlap between nonorthogonal multi-electrons wavefunctions. 
    Key point is to specify excitations in ci1 and ci2
     
    overlapCI:      Overlap between different configurations.
    
    overlapCAS:     Overlap between mcscf wavefunction. 
    '''
    overlapCI = numpy.zeros((len(ci1[0]),len(ci2[0])))
    config = configurations_generator(ncas, nelecas)
    
    for mm in range(len(ci1[0])):
        for nn in range(len(ci2[0])):
            overlapCI[mm][nn] = get_ovlp_nonothorgonal_configurations(overlapMO, nelec, ncas, nelecas, config[mm], config[nn])

    overlapCAS = numpy.zeros((2,2))
    for ii in range(2):
        for jj in range(2):
            overlapCAS[ii][jj] = numpy.einsum("i,ij,j->", ci1[ii], overlapCI, ci2[jj])

    return overlapCAS


def get_ovlp_nonothorgonal_configurations(overlapMO, nelec, ncas, nelecas, config1, config2):
    '''
    Calculate overlap between nonorthogonal configurations. 
    It is based on L\"owdin Formula equ(8) in J. Phys. Chem. Lett. 2015, 6, 4200-4203.
    '''
    nelecore = nelec - nelecas
    detTMP = numpy.zeros((nelec, nelec))

    for ii in range(nelec):
        if(ii < nelecore):
            iitmp = ii
        else:
            iitmp = config1[ii - nelecore] + nelecore
        for jj in range(nelec):
            if(jj < nelecore):
                jjtmp = jj
            else:
                jjtmp = config2[jj - nelecore] + nelecore
            if(divmod(iitmp+jjtmp,2)[1]!=0):
                # Overlap between spin orbitals with different spin is 0.
                detTMP[ii][jj] = 0.0
            else:
                # Turn overlap between spin orbitals to spatial orbitals.
                detTMP[ii][jj] = overlapMO[iitmp//2][jjtmp//2]

    overlapCI = numpy.linalg.det(detTMP)
    return overlapCI

    
def configurations_generator(ncas, nelecas):
    '''
    Generate all possible configurations for active space (nelecas in ncas)
    Return an array with spin orbitals which alpha and beta electrons occupy.
    The range of elements in the array is from 0 to ncas - 1, which is define in active space.
    '''
    nelecas_b = nelecas//2
    nelecas_a = nelecas - nelecas_b

    nRow = int(comb(ncas, nelecas_a))
    nCol = int(comb(ncas, nelecas_b))

    tmp_a = combination_generator(ncas, nelecas_a)
    tmp_b = combination_generator(ncas, nelecas_b)
    for ii in range(tmp_a.shape[0]):
        for jj in range(tmp_a.shape[1]):
            tmp_a[ii][jj] = 2 * tmp_a[ii][jj]
    for ii in range(tmp_b.shape[0]):
        for jj in range(tmp_b.shape[1]):
            tmp_b[ii][jj] = 2 * tmp_b[ii][jj] + 1
    tmp = []
    nn = 0
    for ii in range(nRow):
        for jj in range(nCol):
            tmp.append([])
            for kk in range(tmp_a.shape[1]):
                tmp[nn].append(int(tmp_a[ii][kk]))
            for ll in range(tmp_b.shape[1]):
                tmp[nn].append(int(tmp_b[jj][ll]))
            nn = nn + 1

    return tmp
    

def combination_generator(N, k):
    '''
    Generate C_N^k combinations in the order of ci_coeff 
    using itertools.combinations function. In PySCF, configurations
    are generated in a reverse order of combinations. 

    There is also an older version using recursion function, 
    but it suffers from a quite low speed.
    '''
    tmp1 = numpy.array(list(itertools.combinations(sorted(list(range(N)), reverse = True), k)))
    tmp = numpy.zeros((tmp1.shape[0], tmp1.shape[1]))
    # Turn tmp1 to a reverse order
    for ii in range(tmp1.shape[0]):
        for jj in range(tmp1.shape[1]):
            iitmp = - ii - 1
            jjtmp = - jj - 1
            if(ii == 0):
                iitmp = tmp1.shape[0] - 1
            if(jj == 0):
                jjtmp = tmp1.shape[1] - 1
            tmp[ii][jj] = tmp1[iitmp][jjtmp]

    return tmp
    # NSize = int(comb(N, k))
    # tmp = numpy.zeros((NSize, k))
    # def recursion(tmp, ii, k, n, ncount = [0]):
    #     if(ii < k):
    #         for jj in range(k - ii, n):
    #             for nn in range(ncount[0], tmp.shape[0]):
    #                 tmp[nn][-ii] = n - 1
    #             recursion(tmp, ii + 1, k, n = jj, ncount = ncount)
    #     else:
    #         tmp[ncount[0]][-ii] = n - 1
    #         ncount[0] = ncount[0] + 1  
    # recursion(tmp, 0, k, N+1)
    # return tmp
    
