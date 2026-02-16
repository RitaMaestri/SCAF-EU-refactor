#equations cge
import numpy as np
from math import sqrt
from simple_calibration import A,M,SE,E,ST,CH,T
#import data_calibration_from_matrix as dt

#EQUATIONS
###############################################################################################################

def eqKLj(KLj,bKL, bKLj, Lj, Kj, alphaLj, alphaKj):
    #print("eqKLj")

    zero = -1 + KLj / (
        bKL *bKLj*np.float_power(Lj,alphaLj) * np.float_power(Kj,alphaKj)
    )

    return zero

###############################################################################################################

def eqKLj_E(KLj,bKL, bKLj, Lj, Kj, alphaLj, alphaKj):
    #print("eqKLj")
    bKL_array= np.repeat(bKL, len(bKLj))
    bKL_array[1]=1
    
    zero = -1 + KLj / (
        bKL_array *bKLj*np.float_power(Lj,alphaLj) * np.float_power(Kj,alphaKj)
    )

    return zero


###############################################################################################################

def eqFj(Fj,pF,KLj,pKLj,alphaFj):

    zero= -1 + Fj / (
        (np.prod(np.vstack([pKLj,KLj,alphaFj]), axis=0))/pF
    )

    return zero

###############################################################################################################

def eqlj(l, alphalj, KLj, Lj):
    zero = - 1 + KLj * alphalj * l /Lj
    return zero

###############################################################################################################

def eqYij(Yij,aYij,Yj, _index=None):

    Yjd=np.diag(Yj)
    
    if isinstance(_index, np.ndarray):
        #print("Yij check: ",(Yij[_index[0],_index[1]]==dt.variables['Yijn0']).all())
        
        zero= -1 + Yij[_index[0],_index[1]] / np.dot(aYij,Yjd)[_index[0],_index[1]]
    else:
        zero= -1 + Yij / np.dot(aYij,Yjd)
        zero=zero.flatten()
    #convert matrix to vector

    return zero

###############################################################################################################

def eqLeontiefVolumes(quantity, technical_coeff, output, _index=None):
    """
    Leontief-type equation for vectorial technical coefficients.
    
    Relates a quantity to a technical coefficient and output:
    quantity = technical_coeff * output
    
    When technical_coeff = 0, enforces quantity = 0.
    
    Used for:
    - KL: KLj = aKLj * Yj
    - YE_B: YE_Bj = aYE_Bj * Yj (buildings energy)
    - YE_P: YE_Pj = aYE_Pj * Yj (process energy)
    - YE_T: YE_Tj = aYE_Tj * Yj (transport energy)
    
    Parameters
    ----------
    _index : np.ndarray, optional
        Index array of elements to check. Only applies equation to non-zero elements.
        If not provided, automatically handles zeros via np.where.
    """
    if isinstance(_index, np.ndarray):
        # Apply Leontief equation only to non-zero elements indicated by _index
        zero = -1 + quantity[_index] / np.multiply(technical_coeff[_index], output[_index])
    else:

        zero = -1 + quantity / np.multiply(technical_coeff, output)
    return zero

###############################################################################################################

def eqpYj(pYj,pCj,aKLj,pKLj,aYij, tauYj):

    pCjd=np.diag(pCj)

    zero= -1 + pYj / (
        ( aKLj * pKLj + np.dot(pCjd,aYij).sum(axis=0) )*(1+tauYj) #AXIS=0 sum over the rows CHECKED
    )

    return zero


###############################################################################################################

def eqpYj_E(pYj, pCj, aKLj, pKLj, aYij, pY_Ej, tauYj):
    #creo una matrice (N-1)xN (ha una riga in meno)
    pCjnE = np.delete(pCj, E)
    
    aYijnE = np.delete(aYij, (E), axis=0)
    
    pCjd=np.diag(pCjnE)
    #il risultato è un vettore riga.
    #sommo su tutta la colonna (i costi). 
    #la riga dell'energia ha prezzi eterogenei -> la levo dal dot product e la aggiungo dopo) 
    zero= -1 + pYj / (
        ( aKLj * pKLj + np.dot(pCjd,aYijnE).sum(axis=0) + aYij[E]*pY_Ej )*( 1+tauYj ) #AXIS=0 sum over the rows CHECKED
    )
    
    return zero

###############################################################################################################

def eqCES(Zj, thetaj, alphaXj,alphaYj,Xj,Yj,sigmaj,_index=None):
    
    if isinstance(_index, np.ndarray):
        Xj=Xj[_index]
        Zj=Zj[_index]
        alphaXj=alphaXj[_index]
        alphaYj=alphaYj[_index]
        sigmaj=sigmaj[_index]
        thetaj=thetaj[_index]


    etaj = ( sigmaj-1 ) / sigmaj

    partj = alphaXj * np.float_power(Xj,etaj, out=np.zeros(len(Xj)),where=(Xj!=0)) + alphaYj * np.float_power(Yj,etaj, out=np.zeros(len(Yj)),where=(Yj!=0))

    zero = -1 + Zj / ( np.float_power( partj, 1/etaj ) * thetaj )

    return zero

###############################################################################################################

def eqCESquantity(Xj, Zj, thetaj, alphaXj, alphaYj, pXj, pYj, sigmaj, _index=None, theta=1):
        
    if isinstance(_index, np.ndarray):
        Xj=Xj[_index]
        Zj=Zj[_index]
        alphaXj=alphaXj[_index]
        alphaYj=alphaYj[_index]
        pXj=pXj[_index]
        pYj=pYj[_index]
        sigmaj=sigmaj[_index]
        thetaj=thetaj[_index]

    #is it correct??? TODO
    partj = np.float_power(alphaXj, sigmaj, out=np.zeros(len(alphaXj)),where=(alphaXj!=0)) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj, out=np.zeros(len(alphaYj)),where=(alphaYj!=0))* np.float_power(pYj,1-sigmaj)
    
    zero= -1 + Xj / (
        np.float_power(alphaXj/pXj, sigmaj) * np.float_power(partj , sigmaj/(1-sigmaj) ) * Zj * np.float_power(thetaj*theta,-1)
    )
    return zero

###############################################################################################################

def eqCESprice(pZj,pXj,pYj,alphaXj,alphaYj,sigmaj, thetaj, theta=1, E_exception=False):

    partj= np.float_power(alphaXj,sigmaj, out=np.zeros(len(alphaXj)),where=(alphaXj!=0)) * np.float_power(pXj,1-sigmaj) + np.float_power(alphaYj,sigmaj, out=np.zeros(len(alphaYj)),where=(alphaYj!=0)) * np.float_power(pYj,1-sigmaj)
    
    if E_exception:
        theta_array = np.repeat(theta, len(thetaj))
        theta_array[E] = 1 
        TFP = theta_array * thetaj
    else:
        TFP= theta*thetaj
        
    zero= -1 + pZj / (  
        np.float_power(TFP, -1) *
        np.float_power(partj, 1/(1-sigmaj) )
    )

    return zero

###############################################################################################################

def eqB(B,pXj,Xj,pMj,Mj):
    
    zero = -1 + B / sum(pXj*Xj-pMj*Mj)
    
    return zero

###############################################################################################################

def eqMultiplication(result, mult1, mult2):
    
    zero = -1 + result / (mult1*mult2)
    
    return(zero)

###############################################################################################################

##check for the case where the index is an empty array! TODO
def eqCobbDouglasj(Qj,alphaQj,pCj,Q,_index=None):
    
    if isinstance(_index, np.ndarray):
        zero= -1 + Qj[_index] / ( alphaQj[_index] * (Q/ pCj[_index]) )
    else:
        zero= -1 + Qj / ( alphaQj * (Q/ pCj) )
    
    return zero

###############################################################################################################

def eqalphaCj(alphaCj,R,pCj,alphaCDESj,betaRj, _index=None):
    if isinstance(_index, np.ndarray):
        zero= -1 + alphaCj[_index] / (alphaCDESj * np.float_power( R/pCj , betaRj ) / sum(alphaCDESj * np.float_power( R/pCj , betaRj )))[_index]
    else:
        zero= -1 + alphaCj / (alphaCDESj * np.float_power( R/pCj , betaRj ) / sum(alphaCDESj * np.float_power( R/pCj , betaRj )))

    return zero

###############################################################################################################

def eqR(R,Cj,pCj):

    zero = -1 + R / sum(Cj*pCj)

    return zero / sum(Cj*pCj)

    return zero

###############################################################################################################

def eqTotalConsumptions(pCj, Qj, Q):
    
    zero= - 1 + sum(pCj*Qj)/ Q
    
    return zero

###############################################################################################################

def eqSj(Sj,Cj,Gj,Ij,Yij):
    #print("eqSj")
    zero = -1 + Sj / (
        (Cj + Gj + Ij + Yij.sum(axis=1))#sum over the rows
    )

    return zero

###############################################################################################################
#same equation for L and K

def eqF(F,Fj):
    #print("eqF")

    zero= -1 + F / sum(Fj)

    return zero

###############################################################################################################

def eqID(x,y, _index=None):
    #print("eqID")
    if isinstance(_index, np.ndarray):
        zero=-1 + x[_index] / y[_index]
    else:
        zero=-1 + x / y
    return zero
    


###############################################################################################################

def eqGDP(GDP,pCj,Cj,Gj,Ij,pXj,Xj,pMj,Mj):
    #print("eqGDP")

    zero= -1 + GDP / sum(pCj*(Cj+Gj+Ij)+pXj*Xj-pMj*Mj)

    return zero

###############################################################################################################

def eqGDP_E(GDP,pCj,Cj,Gj,Ij,pXj,Xj,pMj,Mj, pC_EC):
    #print("eqGDP")
    
    mask = np.full(len(pCj), True)
    mask[E] = False

    zero= -1 + GDP / ( np.sum( pCj*(Cj+Gj+Ij), where=mask)+ Cj[E] * pC_EC + sum(pXj*Xj-pMj*Mj) )

    return zero

###############################################################################################################

#tp=time_previous
def eqGDPPI(GDPPI,pCj,pCtp,pXj,pXtp,Cj,Gj,Ij,Xj,Mj,Ctp,Gtp,Itp,Xtp,Mtp):

    zero=-1 + GDPPI / sqrt(
        ( sum( pCj*(Cj+Gj+Ij)+pXj*(Xj-Mj) ) / sum( pCtp*(Cj+Gj+Ij)+pXtp*(Xj-Mj) ) ) * ( sum( pCj*(Ctp+Gtp+Itp)+pXj*(Xtp-Mtp) ) / sum( pCtp*(Ctp+Gtp+Itp)+pXtp*(Xtp-Mtp) ) )
    )
    return zero

###############################################################################################################

def eqCPI(CPI,pCj,pCtp,Cj,Ctp):

    zero=-1 + CPI / sqrt(
        ( sum( pCj*Cj ) / sum( pCtp*Cj ) ) * ( sum( pCj*Ctp ) / sum( pCtp*Ctp ) )
    )

    return zero

###############################################################################################################

#GDPPI is the GDPPI time series
def eqGDPreal(GDPreal, GDP, GDPPI):
   
    zero=-1 + GDPreal /(
        GDP / np.prod(GDPPI)
    )
    return zero

###############################################################################################################

def eqRreal(Rreal, R, CPI):
   
    zero=-1 + Rreal /(
        R / np.prod(CPI)
    )
    return zero

###############################################################################################################

#I put the if  otherwise indexing with None gave me an array of array TODO
def eqCalibi(pX, Xj, data, _index = None):
    #print("eqCalibi")
    if isinstance(_index, np.ndarray):
        zero = -1 + data[_index] / (pX*Xj)[_index] #QUI
    else:
        zero = -1 + data / (pX*Xj)

    return zero

###############################################################################################################

def eqCalibij(pYi, Yij, data, _index=None):
    #print("eqCalibij")
    pYid = np.diag(pYi)
    
    if isinstance(_index, np.ndarray):
        zero = -1 + data[_index[0],_index[1]] / np.dot(pYid,Yij)[_index[0],_index[1]]#QUI

    else:
        zero = -1 + data / np.dot(pYid,Yij) #QUI
        zero=zero.flatten()
        
    return zero

###############################################################################################################

def eqCETquantity(Xj,Yj,alphaXj,pXj,pYj,sigmaj):

    zero = -1 + Xj / (
        np.float_power(alphaXj*pYj/pXj, sigmaj)*Yj
    )

    return zero

###############################################################################################################

def eqsD(sD,Ij,pCj, Mj, Xj, pXj, GDP):
    zero = -1 + sD/(
        sum(Ij*pCj+(Xj-Mj)*pXj)/GDP
        )
    return zero

###############################################################################################################

def eqT(T,tauYj, pYj, Yj, tauSj, pSj, Sj, tauL, w, Lj):

    zero = - 1 + T / (
        
        sum(   tauSj*pSj*Sj + (tauYj/(tauYj+1))*pYj*Yj + tauL*w*Lj  )
        
        )
    return zero

###############################################################################################################

def eqPriceTax(pGross,pNet, tau, exclude_idx=None):
    idx=np.array(range(len(pNet)))
    mask = ~np.isin(idx, exclude_idx)
    idx = idx[mask]
    
    zero = - 1 + (pGross / (
        pNet*(1+tau)
        ))[idx]
    return zero

###############################################################################################################

def eqRi(Ri,sL,w,Lj,sK,Kj,pK,sG,T,Rg,B):

    zero = - 1 + Ri / ( sL*w*sum(Lj) + sK*sum(Kj)*pK + sG*(T-Rg) - B )
    return zero

###############################################################################################################

def eqIneok(I,K,alphaIK):
    
    zero = -1 + I/(alphaIK*K)
    
    return zero

###############################################################################################################

def equ(u,L,Lj):
    zero = - 1 + u / (( L - sum(Lj) ) / L) 
    return zero

###############################################################################################################

def eqw_real(w_real,CPI,w):
    zero=-1 + w_real /(
        w / np.prod(CPI)
    )
    return zero

###############################################################################################################

def eqw_curve(w_real, alphaw, u, sigmaw):
    zero = 1 - w_real / ( alphaw *(u**sigmaw) )
    
    return zero

###############################################################################################################

def eqIj(Ij,alphaIj,I,_index=None):
    
    if isinstance(_index, np.ndarray):
        zero= -1 + Ij[_index] / ( alphaIj[_index] * I)
    else:
        zero= -1 + Ij / ( alphaIj * I )
    
    return zero

###############################################################################################################

def eqpI(pI,pCj,alphaIj):
    zero = 1 - pI / ( sum(pCj*alphaIj) )
    return zero
    
###############################################################################################################

def eqinventory(Knext,K,delta, I):
    zero = 1 - Knext / ( K * (1-delta) + I )
    return zero

###############################################################################################################

#tested
def eqRevenueCost(p1j,p2j,p12j,V1j,V2j,V12j, _index=None):
    if isinstance(_index, np.ndarray):
        p1j=p1j[_index]
        p2j=p2j[_index]
        p12j=p12j[_index]
        V1j=V1j[_index]
        V2j=V2j[_index]
        V12j=V12j[_index]
        
    zero = 1 - p12j*V12j/(p1j*V1j+p2j*V2j)
    return zero

###############################################################################################################

def eqSameRatio(numerator1,numerator2,denominator1,denominator2):
    zero= 1 - numerator1 * denominator2 / (numerator2 * denominator1)
    return zero

###############################################################################################################

def eq_Z_j(A_Cj,betaCj,u_C,gammaCj,pCj,Cj,R0):

    Z_j = A_Cj * betaCj * u_C ** (gammaCj * betaCj) * (pCj / R0) ** betaCj
    
    return Z_j


def eqC_CDE(A_Cj,betaCj,u_C,gammaCj,pCj,Cj,R):

    Z_j = eq_Z_j(A_Cj,betaCj,u_C,gammaCj,pCj,Cj,R)

    alphaCj=Z_j/sum(Z_j)
    zero=1-pCj*Cj/(R*alphaCj)
    return zero

def eq_u_CDE(norm_factor,A_Cj,betaCj,u_C,gammaCj,pCj,Cj,R):
    x_j = A_Cj * u_C ** (gammaCj * betaCj) * (pCj / R) ** betaCj
    zero= 1 - norm_factor/sum(x_j)
    return zero

def eq_R_E(R_E, pC_E, C_E):
    zero = 1 - R_E/(pC_E*C_E)
    return zero

def eq_RH_nE(R, R_E, R_nE):
    zero = 1 - R_nE/(R-R_E)
    return zero





# _____ _   _ _____ ____   ______   __   ____ ___  _   _ ____  _     ___ _   _  ____ 
#| ____| \ | | ____|  _ \ / ___\ \ / /  / ___/ _ \| | | |  _ \| |   |_ _| \ | |/ ___|
#|  _| |  \| |  _| | |_) | |  _ \ V /  | |  | | | | | | | |_) | |    | ||  \| | |  _ 
#| |___| |\  | |___|  _ <| |_| | | |   | |__| |_| | |_| |  __/| |___ | || |\  | |_| |
#|_____|_| \_|_____|_| \_\\____| |_|    \____\___/ \___/|_|   |_____|___|_| \_|\____|
#                                                                                    


def eqsum_arr(tot, *args):
    # Check if all arguments are NumPy arrays and have the same shape
    if all(isinstance(arg, np.ndarray) for arg in args) and all(arg.shape == args[0].shape for arg in args[1:]):
        # If arguments are arrays of the same shape, perform element-wise sum
        result = np.sum(args, axis=0)
    else:
        # If arguments are not arrays of the same shape, print an error message and stop the program
        raise ValueError("Error: Arguments must be NumPy arrays of the same shape.")
    zero = 1 - tot / result  
    
    return zero


def eqsum_scalar(tot, *args):
    this_sum = sum(np.sum(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
    zero = 1 - tot/this_sum
    return zero


def eqsum_pEYE(p_CE, pY_Ej, C_E, Y_Ej, pE_B, C_EB, YE_Bj, pE_Pj, YE_Pj, pE_TnT, pE_TT, C_ET, YE_Tj, pE_Ej, YE_Ej):
    
    p_Ej=np.append(pY_Ej,p_CE)
    Q_Ej=np.append(Y_Ej,C_E)
    
    Q_EB=np.append(YE_Bj,C_EB)
    Q_EP=np.append(YE_Pj,0)
    Q_ET=np.append(YE_Tj,C_ET)
    Q_EE=np.append(YE_Ej,0)
    
    pE_Bj=np.array([float(0)]*(len(Q_Ej)))
    pE_Bj[[SE,-1]]=pE_B
    pE_Pj=np.append(pE_Pj,0)
    pE_Tj=np.array([float(pE_TnT)]*(len(Q_Ej)))
    pE_Tj[T]=pE_TT
    pE_Ej = np.append(pE_Ej,0)
    
    zero = 1 - p_Ej*Q_Ej / (pE_Bj * Q_EB + pE_Tj * Q_ET + pE_Pj * Q_EP + pE_Ej * Q_EE)
    
    return zero



def eqsum_pESE(p_SE, tauSE, S_E,Y_Ej,C_E, pY_Ej, p_CE):
    p_E = np.append(pY_Ej,p_CE)
    Q_E = np.append(Y_Ej,C_E)
    zero = 1 - ( p_SE*S_E*(1+tauSE) ) / sum(p_E*Q_E)
    return zero



def compute_new_idx_E(n_sectors, idx_E, _index):
    zero_indexes = [i for i in range(n_sectors) if i not in _index]
    E_diff = sum(1 for el in zero_indexes if el < idx_E)
    new_E = idx_E - E_diff
    return new_E



def eqCobbDouglasj_lambda(Cj, alphaCj, pCj, R, lambda_E, lambda_nE, _index=None):
    p_CE=pCj[E]
    C_E =  lambda_E * alphaCj[E] * (R / p_CE)
    
    if isinstance(_index, np.ndarray):
        new_idx_E=compute_new_idx_E(n_sectors=len(pCj), idx_E=E, _index=_index)
        
        resultCj = lambda_nE * alphaCj[_index] * (R/ pCj[_index])
        resultCj[new_idx_E] = C_E
        zero= -1 + Cj[_index] / resultCj
        
    else:
        resultCj= lambda_nE * alphaCj * (R/ pCj)
        resultCj[E]=C_E        
        
        zero= -1 + Cj / resultCj

    return zero



def eqlambda_nE(alphaCj,lambda_E, lambda_nE):

    sum_alpha_Cj = sum(value for index, value in enumerate(alphaCj) if index != E)
    
    zero = 1 -  1/ (lambda_nE * sum_alpha_Cj + lambda_E * alphaCj[E])
    
    return zero



def eqaKLj0(aKLj0, aKLj, lambda_KLM):
    zero= 1-aKLj[E]/(aKLj0[E]*lambda_KLM)
    return zero


def eqaYij0(aYij0, aYij, lambda_KLM):
    
    aYiE=np.delete(aYij[:,E], E)
    aYiE0=np.delete(aYij0[:,E], E)
    
    aYiE_adj=lambda_KLM*aYiE0
    zero= 1-aYiE/aYiE_adj
    return zero


def eqrho(pEi, p_EE, rho, _index=None):
    if isinstance(_index, np.ndarray):
        zero= 1-rho[_index]/(pEi[_index]/p_EE)      #print("Yij check: ",(Yij[_index[0],_index[1]]==dt.variables['Yijn0']).all())
    else:    
        zero= 1-rho/(pEi/p_EE)
    return zero



