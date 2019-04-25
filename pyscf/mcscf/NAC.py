import numpy
from scipy.special import comb

class Config(object):
    #Class for Configurations
    def __init__(self):
        self.excited_a = []
        self.excited_r = []
        if(len(self.excited_a)!=len(self.excited_r)):
            print("ERROR! Number of electrons in this Conguration is WRONG.")
            print("With ", len(self.excited_a), " electrons excited to ", len(self.excited_r), " sites")

    def excited(self, a, r):
        self.excited_a.append(a)
        self.excited_r.append(r)


def get_NAC(mc1, mc2, homoNumber, ncas, nelecas, overlapAllAO, dtheta, nstate1=0, nstate2=1):
    overlapAO = overlapAllAO[:len(overlapAllAO)/2, len(overlapAllAO)/2:]
    tmp = ncas - nelecas/2
    overlapMO = numpy.zeros((homoNumber+tmp, homoNumber+tmp))
    
    for ii in range(homoNumber+tmp):
        for jj in range(homoNumber+tmp):
            moTmp1 = mc1.mo_coeff[..., ii]
            moTmp2 = mc2.mo_coeff[..., jj]
            overlapMO[ii][jj] = numpy.einsum("i,ij,j->", moTmp1, overlapAO, moTmp2)

    ci1 = numpy.zeros((2,len(mc1.ci[nstate1])*len(mc1.ci[nstate1])))
    ci2 = numpy.zeros((2,len(mc2.ci[nstate2])*len(mc2.ci[nstate2])))
    for ii in range(len(ci1[0])):
        ci1[0][ii] = mc1.ci[nstate1][ii//5][ii-ii//5*5]
        ci2[0][ii] = mc2.ci[nstate1][ii//5][ii-ii//5*5]
        ci1[1][ii] = mc1.ci[nstate2][ii//5][ii-ii//5*5]
        ci2[1][ii] = mc2.ci[nstate2][ii//5][ii-ii//5*5]
    
    overlapCAS = get_ovlp_nonothorgonal(ci1, ci2, ncas, nelecas, homoNumber, overlapMO)
    NAC = (overlapCAS[1][0]-overlapCAS[0][1])/4.0/dtheta/3.14159265*180.0

    return NAC, overlapCAS


def get_ovlp_nonothorgonal(ci1, ci2, ncas, nelecas, homoNumber, overlapMO):
    overlapCI = numpy.zeros((len(ci1[0]),len(ci2[0])))
    tmp = []
    for ii in range(25):
        tmp.append(Config())

    for ii in range(5):
        for jj in range(5):
            tmp[ii*5+jj].excited(1, ii*2+1)
            tmp[ii*5+jj].excited(0, jj*2)
            #print (tmp[ii*5+jj].excited_a)
            #print (tmp[ii*5+jj].excited_r)

    #############TODO
    for mm in range(len(ci1[0])):
        for nn in range(len(ci2[0])):
            overlapCI[mm][nn] = get_ovlp_nonothorgonal_configurations(overlapMO, homoNumber, ncas, nelecas, tmp[mm], tmp[nn])

    print(overlapCI)
    overlapCAS = numpy.zeros((2,2))
    for ii in range(2):
        for jj in range(2):
            overlapCAS[ii][jj] = numpy.einsum("i,ij,j->", ci1[ii], overlapCI, ci2[jj])

    return overlapCAS


def get_ovlp_nonothorgonal_configurations(overlapMO, homoNumber, ncas, nelecas, config1, config2):
    nelecore = homoNumber*2 - nelecas
    detTMP = numpy.zeros((homoNumber*2, homoNumber*2))

    iitmp = 0
    jjtmp = 0
    for ii in range(homoNumber*2):
        iitmp = ii
        for mm in range(len(config1.excited_a)):
            if(ii==(nelecore + config1.excited_a[mm])):
                iitmp = nelecore + config1.excited_r[mm]                  
        for jj in range(homoNumber*2):
            jjtmp = jj
            for mm in range(len(config2.excited_a)):
                if(jj==(nelecore + config2.excited_a[mm])):
                    jjtmp = nelecore + config2.excited_r[mm]                    
            if(divmod(iitmp+jjtmp,2)[1]!=0):
                detTMP[ii][jj] = 0.0
            else:
                detTMP[ii][jj] = overlapMO[iitmp//2][jjtmp//2]

    overlapCI = numpy.linalg.det(detTMP)
    return overlapCI

    
