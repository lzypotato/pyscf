import numpy
from scipy.special import comb


def numerical_NAC(mc1, mc2, nelec, ncas, nelecas, overlapAllAO, dR, nstate1=0, nstate2=1):
    '''
    Numerical non-adiabatic-coupling calculator
    '''
    overlapAO = overlapAllAO[:len(overlapAllAO)/2, len(overlapAllAO)/2:]
    nelecore = nelec - nelecas
    overlapMO = numpy.zeros((nelecore//2+ncas, nelecore//2+ncas))
    #dR = dR/3.14159265*180.0  # If the unit of dR is degree, it's better to convert it to RAD
    
    for ii in range(nelecore//2+ncas):
        for jj in range(nelecore//2+ncas):
            moTmp1 = mc1.mo_coeff[..., ii]
            moTmp2 = mc2.mo_coeff[..., jj]
            overlapMO[ii][jj] = numpy.einsum("i,ij,j->", moTmp1, overlapAO, moTmp2)

    ciRow = mc1.ci[nstate1].shape[0]
    ciColumn = mc1.ci[nstate1].shape[1]

    # ci is one-dimensional array converted from ci matrix from mcscf calculations
    ci1 = numpy.zeros((2,ciRow*ciColumn))
    ci2 = numpy.zeros((2,ciRow*ciColumn))
    for ii in range(len(ci1[0])):
        ci1[0][ii] = mc1.ci[nstate1][ii//ciColumn][ii-ii//ciColumn*ciColumn]
        ci2[0][ii] = mc2.ci[nstate1][ii//ciColumn][ii-ii//ciColumn*ciColumn]
        ci1[1][ii] = mc1.ci[nstate2][ii//ciColumn][ii-ii//ciColumn*ciColumn]
        ci2[1][ii] = mc2.ci[nstate2][ii//ciColumn][ii-ii//ciColumn*ciColumn]
    
    overlapCAS = get_ovlp_nonothorgonal(ci1, ci2, ncas, nelecas, nelec, overlapMO)
    NAC = (overlapCAS[1][0]-overlapCAS[0][1])/4.0/dR

    return NAC, overlapCAS


def get_ovlp_nonothorgonal(ci1, ci2, ncas, nelecas, nelec, overlapMO):
    '''
    Calculate overlap between nonorthogonal multi-electrons wavefunctions. 
    Key point is to specify excitations in ci1 and ci2
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
    config1 and config2 belong to class Config. 
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
                detTMP[ii][jj] = 0.0
            else:
                detTMP[ii][jj] = overlapMO[iitmp//2][jjtmp//2]

    overlapCI = numpy.linalg.det(detTMP)
    return overlapCI

    
def configurations_generator(ncas, nelecas):
    '''
    Generate all configurations for active space (nelecas in ncas)
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
    Generate C_N^k combinations in the order of ci_coeff using recursive function.
    '''
    NSize = int(comb(N, k))
    tmp = numpy.zeros((NSize, k))

    def recursion(tmp, ii, k, n, ncount = [0]):
        if(ii < k):
            for jj in range(k - ii, n):
                for nn in range(ncount[0], tmp.shape[0]):
                    tmp[nn][-ii] = n - 1
                recursion(tmp, ii + 1, k, n = jj, ncount = ncount)
        else:
            tmp[ncount[0]][-ii] = n - 1
            ncount[0] = ncount[0] + 1
    
    recursion(tmp, 0, k, N+1)
    return tmp
