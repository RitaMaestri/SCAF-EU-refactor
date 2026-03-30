import numpy as np
import helpers.model_equations as eq
from import_EXIOBASE import non_zero_index_G, non_zero_index_I, non_zero_index_X, non_zero_index_M, non_zero_index_Yij, non_zero_index_C
from calibration import SE, E, N

def fill_nans(par_value, var_value, key):
    if isinstance(par_value, np.ndarray):
        par_copy = par_value.copy()
        if par_copy.ndim == 2:
            mask_par = np.isnan(par_copy)
            row_idx, col_idx = np.where(mask_par)
            par_copy[row_idx, col_idx] = var_value.flatten()
        else:
            mask_par = np.isnan(par_copy)
            par_copy[mask_par] = var_value
        return par_copy
    else:
        return var_value


def joint_dict(par, var):
    # Create a new dictionary to store the updated values
    d = par.copy()
    # Iterate through keys of A that are also present in B
    for key in var.keys() & par.keys():
        if np.isscalar(var[key]):
            d[key] = fill_nans(par[key], np.array([var[key]]), key)
        else:
            d[key] = fill_nans(par[key], var[key],key)
    return d

def find_keys_with_large_elements(dictionary, threshold=10e-5):
    keys_with_large_elements = []

    for key, value in dictionary.items():
        # Check if at least one element in the array is greater than one
        if isinstance(value, (list, np.ndarray)):
            if np.any(abs(np.array(value)) > threshold):
                keys_with_large_elements.append(key)

    return keys_with_large_elements

def system(var, par):

    d=joint_dict(par,var)
    ## for computational reasons, we need to exclude the equations that involve variables that are zero 
    # in the initial calibration, as they would cause the solver to diverge. 
    # the solver checks if the number of equations is the same as the number of NON ZERO endogenousvariables
    non_zero_index_intermediate_E_vol = np.where(d["E_vol"][:-1, :] != 0) #where the energy volumes are not zero in the matrix w/o households
    index_wo_E=np.delete(np.array(range(N)), E)
    index_E=np.array([E])
    non_zero_index_C_wo_E=np.intersect1d(index_wo_E, non_zero_index_C)
    index_wo_E_SE=np.delete(index_wo_E, SE)
    global equations
    # ADD NON ZERO INDEXES FOR ENERGY MATRICES
    equations= {
        ###
        "eqCESquantityKj":eq.eqCESquantity(Xj=d['Kj'], Zj=d['KLj'], alphaXj=d['alphaKj'], alphaYj=d['alphaLj'], pXj=d['pK'], pYj=d['pL'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
        ###
        "eqCESpriceKL":eq.eqCESprice(pZj=d['pKLj'], pXj=d['pL'], pYj=d['pK'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta = d['bKL'], E_exception=True),
        ###
        "eqYij":eq.eqYij(Yij=d['Yij'], aYij=d['aYij'],Yj=d['Yj'], _index=non_zero_index_Yij),
        ###
        "eqLeontiefVolumes_KL":eq.eqLeontiefVolumes(quantity=d['KLj'],technical_coeff=d['aKLj'],output=d['Yj']),
        ###
        ### da verificare uguaglianza
        ### 
        "eqpYj_E":eq.eqpYj_E(pYj=d['pYj'], pCj=d['pCj'], aKLj=d['aKLj'], pKLj=d['pKLj'], aYij=d['aYij'], pY_Ej=d["pY_Ej"], tauYj=d['tauYj']),
        ###
        "eqCESquantityX":eq.eqCESquantity(Xj=d['Xj'], Zj=d['Yj'] , alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], pXj=d['pXj'], pYj=d['pDj'], sigmaj=d['sigmaXj'], thetaj=d['thetaj'], _index=np.intersect1d(index_wo_E,non_zero_index_X)),
        ###
        "eqCESquantityDy":eq.eqCESquantity(Xj=d['Dj'], Zj=d['Yj'], alphaXj=d['alphaDj'], alphaYj=d['alphaXj'], pXj=d['pDj'], pYj=d['pXj'], sigmaj=d['sigmaXj'],  thetaj=d['thetaj'], _index=index_wo_E),
        ###
        "eqDy_Equantity":eq.eqsum_scalar(d['Yj'][E], d['Dj'][E], d['Xj'][E]),
        ###
        "eqpriceYj":eq.eqRevenueCost(p1j=d['pDj'],p2j=d['pXj'],p12j=d['pYj'],V1j=d['Dj'],V2j=d['Xj'],V12j=d['Yj']),
        ###
        "eqCESquantityDs":eq.eqCESquantity(Xj=d['Dj'], Zj=d['Sj'], alphaXj=d['betaDj'], alphaYj=d['betaMj'], pXj=d['pDj'], pYj=d['pMj'], sigmaj=d['sigmaSj'], thetaj=d['csij']),
        ###
        #"eqDs_Equantity":eq.eqsum_scalar(d['Sj'][E], d['Dj'][E], d['Mj'][E]),
        ###
        "eqCESquantityM":eq.eqCESquantity(Xj=d['Mj'], Zj=d['Sj'], alphaXj=d['betaMj'], alphaYj=d['betaDj'], pXj=d['pMj'], pYj=d['pDj'], sigmaj=d['sigmaSj'], thetaj=d['csij'], _index=non_zero_index_M),#np.intersect1d(index_wo_E,non_zero_index_M)
        ###
        "eqpriceSj":eq.eqRevenueCost(p1j=d['pDj'],p2j=d['pMj'],p12j=d['pSj'],V1j=d['Dj'],V2j=d['Mj'],V12j=d['Sj']),
        ###
        "eqB":eq.eqB(B=d['B'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        ###
        "eqIDpX":eq.eqID(x=d['pXj'],y=d['pMj'],_index=index_wo_E),
        ###
        "eqCobbDouglasjG":eq.eqCobbDouglasj(Qj=d['Gj'],alphaQj=d['alphaGj'],pCj=d['pCj'],Q=d['Rg'], _index=non_zero_index_G),
        ###
        "eqIj":eq.eqIj(Ij=d['Ij'],alphaIj=d['alphaIj'],I=d['I'],_index=non_zero_index_I),
        ###
        "eqMultRg":eq.eqMultiplication(result=d['Rg'],mult1=d['wG'],mult2=d['GDP']),
        ###
        "eqSj":eq.eqSj(Sj=d['Sj'],Cj=d['Cj'], Gj=d['Gj'], Ij=d['Ij'], Yij=d['Yij']),
        ###
        "eqGDP":eq.eqGDP(GDP=d['GDP'],pCj=d['pCj'],Cj=d['Cj'],Gj=d['Gj'],Ij=d['Ij'],pXj=d['pXj'],Xj=d['Xj'],pMj=d['pMj'],Mj=d['Mj']),
        ###
        "eqGDPPI":eq.eqGDPPI(GDPPI = d['GDPPI'], pCj=d['pCj'], pXj=d['pXj'], pCtp= d['pCtp'], pXtp=d['pXtp'], Cj= d['Cj'], Gj= d['Gj'], Ij= d['Ij'], Xj=d['Xj'], Mj=d['Mj'], Ctp= d['Ctp'], Gtp= d['Gtp'], Itp= d['Itp'], Xtp=d['Xtp'], Mtp=d['Mtp']),
        ###
        "eqGDPreal":eq.eqGDPreal(GDPreal=d['GDPreal'],GDP=d['GDP'], GDPPI=d['GDPPI']), #expected GDPPI time series
        ###
        "eqPriceTaxtauS":eq.eqPriceTax(pGross=d['pCj'], pNet=d['pSj'], tau=d['tauSj'], exclude_idx=E),
        ###
        "eqpI":eq.eqpI(pI=d['pI'],pCj=d['pCj'],alphaIj=d['alphaIj']),
        ###
        "eqMultRi":eq.eqMultiplication(result=d['Ri'],mult1=d['pI'],mult2=d['I']),
        ###


        #energy coupling
        ###
    
        "eqEnergyVolumes":eq.eqEnergyVolumesSum(E_matrix=d["E_vol"],Y_Ej=d['Yij'][E,:],C_E=d['Cj'][E]),
        
        "eqEnergyValues":eq.eqEnergyPrices(E_matrix=d["E_vol"],Y_Ej=d['Yij'][E,:],C_E=d['Cj'][E],pE_matrix=d["pE"],pY_Ej=d['pY_Ej'],p_CE=d['pCj'][E]),
        
        "eqRhos":eq.eqRhos(pE=d["pE"], rhos=d["rhoE"]),

        "eqaEj":eq.eqaEj(a_Eej=d["a_Ej"], E_vol=d["E_vol"], Yj=d['Yj'], _index = non_zero_index_intermediate_E_vol),

        "eqPrimaryEnergyPrices":eq.eqPricePrimaryEnergy(pE=d["pE"], energy_index=E),
        
        "eqRhoX":eq.eqRhoTrade(rho=d["rhoX"], pE_trade=d['pXj'][E], pE=d["pE"]),

        "eqRhoM":eq.eqRhoTrade(rho=d["rhoM"], pE_trade=d['pMj'][E], pE=d["pE"]),
        ### Cobb Douglas
        "eqCobbDouglasjC":eq.eqCobbDouglasj(Qj=d['Cj'][non_zero_index_C_wo_E],alphaQj=d['alphaCj_nE'],pCj=d['pCj'][non_zero_index_C_wo_E],Q=d['R_nE']),

        ### CDES
        #"eqCj_CDE":eq.eqC_CDE(A_Cj=d["A_Cj_nE"],betaCj=d["betaCj_nE"],u_C=d["u_C"],gammaCj=d["gammaCj_nE"],pCj=np.delete(d["pCj"], E),Cj=np.delete(d["Cj"], E),R=d["R_nE"]),
        ###
        #"eq_u_CDE":eq.eq_u_CDE(norm_factor=d["normalisation_factor"], A_Cj=d["A_Cj_nE"],betaCj=d["betaCj_nE"],u_C=d["u_C"],gammaCj=d["gammaCj_nE"],pCj=np.delete(d["pCj"], E),Cj=np.delete(d["Cj"], E),R=d["R_nE"]),
        #households energy budget
        ###
       ###
        #"eq_u_CDE":eq.eq_u_CDE(norm_factor=d["normalisation_factor"], A_Cj=d["A_Cj_nE"],betaCj=d["betaCj_nE"],u_C=d["u_C"],gammaCj=d["gammaCj_nE"],pCj=np.delete(d["pCj"], E),Cj=np.delete(d["Cj"], E),R=d["R_nE"]),      "eq_R_E":eq.eq_R_E(R_E=d["R_E"], pC_E=d["pCj"][E], C_E=d["Cj"][E]),
        ###
        "eq_R_E":eq.eq_R_E(R_E=d["R_E"], pC_E=d["pCj"][E], C_E=d["Cj"][E]),
        ###
        "eq_RH_nE":eq.eq_RH_nE(R=d["R"], R_E=d["R_E"], R_nE=d["R_nE"]),
        ###
        "eqpCE":eq.eqsum_pESE(p_SE=d['pSj'][E], tauSE=d['tauSj'][E], S_E=d['Sj'][E], Y_Ej=d['Yij'][E,:], C_E=d['Cj'][E], pY_Ej=d['pY_Ej'], p_CE=d['pCj'][E]),#
        #lambdaKLM


        ###
        "eqaKLj0":eq.eqaKLj0(aKLj0=d['aKLj0'], aKLj=d['aKLj'], lambda_KLM=d['lambda_KLM']),
        ###
        'eqaYij0':eq.eqaYij0(aYij0=d['aYij0'], aYij=d['aYij'], lambda_KLM=d['lambda_KLM']),#
        ###

        "eqWorldPrices": eq.eqSameRatio(numerator1=d['pXj'][index_wo_E_SE],numerator2=d['pYj'][index_wo_E_SE],denominator1=d['pXj'][SE],denominator2=d['pYj'][SE]),
        ###
        "eqMultwI":eq.eqMultiplication(result=d['Ri'],mult1=d['wI'],mult2=d['GDP']),
        ###
        "eqCESquantityLj":eq.eqCESquantity(Xj=d['Lj'], Zj=d['KLj'], alphaXj=d['alphaLj'], alphaYj=d['alphaKj'], pXj=d['pL'], pYj=d['pK'], sigmaj=d['sigmaKLj'], thetaj=d['bKLj'], theta=d['bKL']),#e-5
        ###
        "eqFL":eq.eqF(F=d['L'],Fj=d['Lj']),
        ###
        "eqFK":eq.eqF(F=d['K'],Fj=d['Kj']),
        ###
        "eqMultB":eq.eqMultiplication(result=d['B'],mult1=d['wB'],mult2=d['GDP'])
        

        #"eqAlphaX":eq.eqsum_arr(d['alphaXj'][[ST,CH]], d['alphaXj0'][[ST,CH]], d['lambda_XMj'][[ST,CH]]  ),
        
        #"eqAlphaD":eq.eqsum_arr(d['alphaDj'][[ST,CH]], d['alphaDj0'][[ST,CH]], -d['lambda_XMj'][[ST,CH]]  ),
        
        #"eqBetaD":eq.eqsum_arr(d['betaDj'][[ST,CH]], d['betaDj0'][[ST,CH]], d['lambda_XMj'][[ST,CH]]  ),
        
        #"eqBetaM":eq.eqsum_arr(d['betaMj'][[ST,CH]], d['betaMj0'][[ST,CH]], -d['lambda_XMj'][[ST,CH]]  ),

        #"eqPriceTaxtauL":eq.eqPriceTax(pGross=d['pL'], pNet=d['w'], tau=d['tauL']),

        #"eqCPI":eq.eqCPI(CPI = d['CPI'], pCj=d['pCj'], pCtp= d['pCtp'], Cj= d['Cj'], Ctp= d['Ctp']),
        
        #"eqRreal":eq.eqRreal(Rreal=d['Rreal'],R=d['R'], CPI=d['CPI']), 
        
        #"eqT":eq.eqT(T=d['T'], tauYj=d['tauYj'], pYj=d['pYj'], Yj=d['Yj'], tauSj=d['tauSj'], pSj=d['pSj'], Sj=d['Sj'], tauL=d['tauL'], w=d['w'], Lj=d['Lj']),#
        
        #"eqpYj":eq.eqpYj(pYj=d['pYj'],pCj=d['pCj'],aKLj=d['aKLj'],pKLj=d['pKLj'],aYij=d['aYij'], tauYj=d['tauYj']),
        
        #"eqCESpriceY":eq.eqCESprice(pZj=d['pYj'], pXj=d['pXj'], pYj=d['pDj'], alphaXj=d['alphaXj'], alphaYj=d['alphaDj'], sigmaj=d['sigmaXj'],  thetaj=d['thetaj']),
        
        #"eqCESpriceS":eq.eqCESprice(pZj=d['pSj'],pXj=d['pMj'],pYj=d['pDj'],alphaXj=d['betaMj'],alphaYj=d['betaDj'],sigmaj=d['sigmaSj'], thetaj=d['csij']),
                
        #"eqCobbDouglasjC":eq.eqCobbDouglasj(Qj=d['Cj'],alphaQj=d['alphaCj'],pCj=d['pCj'],Q=d['R'], _index=non_zero_index_C),
        
        # "eqE_P":eq.eqsum_scalar(d['E_P'], d['YE_Pj']),
        
        # "eqE_B":eq.eqsum_scalar(d['E_B'], d['YE_Bj'], d['C_EB']),
        
        # "eqE_T":eq.eqsum_scalar(d['E_T'], d['YE_Tj'], d['C_ET']),
        
        #"eqCj_new":eq.eqCobbDouglasj_lambda(Cj=d['Cj'], alphaCj=d['alphaCj0'],pCj=d['pCj'], R=d['R'], lambda_E=d["lambda_E"], lambda_nE=d["lambda_nE"], _index=non_zero_index_C),#checked
        
        #"eqlambda_nE":eq.eqlambda_nE(alphaCj=d['alphaCj0'],lambda_E=d['lambda_E'], lambda_nE=d['lambda_nE']), #checked

        }
    
    solution = np.hstack(list(equations.values()))
        
    non_zero_equations=find_keys_with_large_elements(equations)

    return solution