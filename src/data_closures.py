import numpy as np 
from simple_calibration import calibrationVariables, N
from import_GTAP_data import sectors
import pandas as pd
import sys
from itertools import product

endo_aYij_indexes = [("ENERGY",x) for x in sectors]
sectors_nE = [x for x in sectors if x != "ENERGY"]
endo_aYij_indexes.extend([(x,"ENERGY") for x in sectors_nE])

class endo_exo_indexes:
    def full_endo(self):
        return []
        
    def full_exo(self):
        return slice(None)
    
    def idx_1D(self, exo_names=None, endo_names=None):
        if (not exo_names == None) and endo_names==None:
            indices_list = [index for index, value in enumerate(self.sectors_names) if value in exo_names]
        elif (not endo_names==None) and exo_names == None:
            indices_list = [index for index, value in enumerate(self.sectors_names) if value not in endo_names]
        else:
            print("wrong arguments for idx_1D")
            sys.exit()
        return indices_list



    #takes couple of sectors names: first name is the row identifier, second name is the column identifier. expected [(sec1,sec2),(sec1,sec3),(sec2,sec4)]   
    def idx_2D(self, exo_names=None, endo_names=None):
        
        if (not exo_names == None) and endo_names==None:
            indexes_list = [(self.sectors_names.index(row), sectors.index(col)) for row, col in exo_names]
            # Sort the indexes_list based on the first element (index_a) and then the second element (index_b)
            sorted_indexes_list = sorted(indexes_list, key=lambda x: (x[0], x[1]))
            rows,cols=zip(*sorted_indexes_list)

        elif (not endo_names==None) and exo_names == None:
            indexes_list = [(sectors.index(row), sectors.index(col)) for row, col in endo_names]
            matrix_size = len(self.sectors_names)
            all_indexes_set = set(product(range(matrix_size), repeat=2))
            # Create a set of present indexes (row_index, column_index) pairs
            present_indexes_set = set(indexes_list)
            # Find the complementary set of indexes (not present in the matrix)
            complementary_indexes_set = all_indexes_set.difference(present_indexes_set)
            # Sort the complementary indexes to maintain order
            sorted_indexes_list = sorted(list(complementary_indexes_set))
        else:
            print("wrong arguments for idx_2D")
            sys.exit()
        
        rows,cols=zip(*sorted_indexes_list)
        return [list(rows), list(cols)]


#######  Class Variable  #############

class Variable(endo_exo_indexes):
    
    def __init__(self, exo_indexes, value):
        
        if isinstance(value, np.ndarray) and value.size == 1:
            self.value=value.item()

        else:
            self.value=value
        
        if isinstance(value, np.ndarray):
            msk = np.zeros(value.shape, dtype=np.bool)
            msk[exo_indexes] = True
            self.exo_mask = msk
            self.endo_mask = ~msk
            
        else:
        #I convert my exo_indexes to a bolean mask.
            self.exo_mask= bool(exo_indexes == self.full_exo())
            self.endo_mask= not self.exo_mask

    def __call__(self):
            print(list([self.exo_mask, self.value]))



######## Class variableSystem ###########


class calibrationDict(endo_exo_indexes):
    
    def assignClosure(self,cal):
        
        sKLneoclassic = (cal.Ri0+cal.B0)/(cal.w * sum(cal.Lj0) + cal.pK0 * sum(cal.Kj0))
        
        sLkaldorian = (cal.Ri0+cal.B0)/(cal.w*sum(cal.Lj0))
        
        
        if self.closure == "johansen":
            return {**self.commonDict, 
                         **{#'sD':Variable(self.full_endo(), cal.sD0),
                            
                            'K':Variable(self.full_exo(), cal.K0),
                            'wI':Variable(self.full_exo(),cal.wI),
                            'wB':Variable(self.full_exo(), cal.wB),
                            'L':Variable(self.full_exo(), cal.L0)
                            }
                         }

        elif self.closure == "neoclassic":
            return {**self.commonDict, 
                         **{'K':Variable(self.full_exo(), cal.K0),
                            'sK':Variable(self.full_exo(), sKLneoclassic),
                            'sL':Variable(self.full_exo(), sKLneoclassic),
                            'sG':Variable(self.full_exo(), 0),
                            'wB':Variable(self.full_exo(), cal.wB),
                            'L':Variable(self.full_exo(), cal.L0)}
                         }

        elif self.closure == "kaldorian":
            return {**self.commonDict, 
                         **{'l':Variable(self.full_endo(), cal.l0),
                            
                            'K':Variable(self.full_exo(), cal.K0),
                            'alphalj':Variable(self.full_exo(), cal.alphalj),
                            'sK':Variable(self.full_exo(), 0),
                            'sL':Variable(self.full_exo(), sLkaldorian),
                            'sG':Variable(self.full_exo(), 0),
                            'wI':Variable(self.full_exo(),cal.wI),
                            'wB':Variable(self.full_exo(), cal.wB),
                            'L':Variable(self.full_exo(), cal.L0)
                            }
                         }

        elif self.closure == "keynes-marshall":
            return {**self.commonDict, 
                         **{'K':Variable(self.full_exo(), cal.K0),
                            'sK':Variable(self.full_exo(), sKLneoclassic),
                            'sL':Variable(self.full_exo(), sKLneoclassic),
                            'sG':Variable(self.full_exo(), 0),
                            'wB':Variable(self.full_exo(), cal.wB),
                            'wI':Variable(self.full_exo(),cal.wI),
                                  
                            }
                         }
        
        elif self.closure == "keynes-kaldor":
            return {**self.commonDict, 
                         **{ 'l':Variable(self.full_endo(), cal.l0),
                            'w_real':Variable(self.full_endo(),cal.w),
                            'uL':Variable(self.full_endo(),cal.uL0),
                            
                            'K':Variable(self.full_exo(), cal.K0),
                            'alphalj':Variable(self.full_exo(), cal.alphalj),
                            'sK':Variable(self.full_exo(), 0),
                            'sL':Variable(self.full_exo(), sLkaldorian),
                            'sG':Variable(self.full_exo(), 0),
                            'wI':Variable(self.full_exo(),cal.wI),
                            'alphaw':Variable(self.full_exo(), cal.alphaw),
                            'sigmaw':Variable(self.full_exo(), cal.sigmaw),
                            'wB':Variable(self.full_exo(), cal.wB),
                            'L':Variable(self.full_exo(), cal.L0u)
                            
                            }
                         }
        
        elif self.closure == "keynes":
            return {**self.commonDict, 
                         **{ 'l':Variable(self.full_endo(), cal.l0),
                            'w_real':Variable(self.full_endo(),cal.w),
                            'uL':Variable(self.full_endo(),cal.uL0),
                            
                            'K':Variable(self.full_exo(), cal.K0),
                            'alphalj':Variable(self.full_exo(), cal.alphalj),
                            'sK':Variable(self.full_exo(), sKLneoclassic),
                            'sL':Variable(self.full_exo(), sKLneoclassic),
                            'sG':Variable(self.full_exo(), 0),
                            'wI':Variable(self.full_exo(),cal.wI),
                            'alphaw':Variable(self.full_exo(), cal.alphaw),
                            'sigmaw':Variable(self.full_exo(), cal.sigmaw),
                            'wB':Variable(self.full_exo(), cal.wB),
                            'L':Variable(self.full_exo(), cal.L0u)
                            }
                         }
        
        elif self.closure == "neokeynesian1":
            return {**self.commonDict, 
                         **{'uL':Variable(self.full_endo(),cal.uL0),
                            'pK_real':Variable(self.full_endo(),cal.pK0),
                            'uK':Variable(self.full_endo(),cal.uK0),
                            'w_real':Variable(self.full_endo(),cal.w),
                             
                             
                            'sK':Variable(self.full_exo(), sKLneoclassic),
                            'sL':Variable(self.full_exo(), sKLneoclassic),
                            'sG':Variable(self.full_exo(), 0),                            
                            'alphaw':Variable(self.full_exo(), cal.alphaw),
                            'sigmaw':Variable(self.full_exo(), cal.sigmaw),
                            'L':Variable(self.full_exo(), cal.L0u),
                            'alphapK':Variable(self.full_exo(), cal.alphapK),
                            'sigmapK':Variable(self.full_exo(), cal.sigmapK),
                            'K':Variable(self.full_exo(), cal.K0u),
                            'wB':Variable(self.full_exo(), cal.wB),
                            
                            }
                    }

        elif self.closure == "neokeynesian2":
            return {**self.commonDict, 
                         **{'w_real':Variable(self.full_endo(),cal.w),
                            'uL':Variable(self.full_endo(),cal.uL0),
                            'pK_real':Variable(self.full_endo(),cal.pK0),
                            'uK':Variable(self.full_endo(),cal.uK0),
                            
                            'sK':Variable(self.full_exo(), sKLneoclassic),
                            'sL':Variable(self.full_exo(), sKLneoclassic),
                            'sG':Variable(self.full_exo(), 0),                            
                            'alphaw':Variable(self.full_exo(), cal.alphaw),
                            'sigmaw':Variable(self.full_exo(), cal.sigmaw),                            
                            'L':Variable(self.full_exo(), cal.L0u),
                            'alphapK':Variable(self.full_exo(), cal.alphapK),
                            'sigmapK':Variable(self.full_exo(), cal.sigmapK),
                            'K':Variable(self.full_exo(), cal.K0u),
                            'alphaIK':Variable(self.full_exo(), cal.alphaIK),

                            }
                    }
        else: 
            print("this closure doesn't exist")
            sys.exit()
    
    def to_endo_dict(self):
        result_dict = {}
        for key, variable in self.variables_dict.items():
            if isinstance(variable.value, np.ndarray):
                masked_array = variable.value[variable.endo_mask]
                if masked_array.size > 0:
                    result_dict[key] = masked_array
            elif variable.endo_mask:
                result_dict[key]=variable.value  
        
        return result_dict
    
    
    def to_exo_dict(self):
        result_dict = {}
        for key, variable in self.variables_dict.items():
            result_dict[key]=variable.value
            if isinstance(variable.value, np.ndarray):
                result_dict[key][variable.endo_mask]=float("nan")  
            elif variable.endo_mask:
                    result_dict[key] = float("nan")  
        return result_dict
    

    
    def __init__(self, closure,initial_L_gr, endoKnext):
        
        self.sectors_names = sectors
        
        cal = calibrationVariables(initial_L_gr)
        self.commonDict = {
            'pL': Variable(self.full_endo(), cal.pL0),
            'pK':Variable(self.full_endo(), cal.pK0),
            'pI':Variable(self.full_endo(), cal.pI0),
            'B':Variable(self.full_endo(), cal.B0),
            'R':Variable(self.full_endo(), cal.R0),
            'Ri':Variable(self.full_endo(), cal.Ri0),
            'Rg':Variable(self.full_endo(), cal.Rg0),
            'bKL':Variable(self.full_endo(), cal.bKL),
            #'CPI':Variable(self.full_endo(), cal.CPI),
            #'Rreal':Variable(self.full_endo(), cal.R0),
            'GDPPI':Variable(self.full_endo(), cal.GDPPI),
            'GDP':Variable(self.full_endo(), cal.GDPreal),
            #'T':Variable(self.full_endo(), cal.T0),
            #'lambda_KLM':Variable(self.idx_1D(endo_names = ["ENERGY"]), cal.lambda_KLM),
            

            'Kj':Variable(self.full_endo(), cal.Kj0),
            'Lj':Variable(self.full_endo(), cal.Lj0),
            'KLj':Variable(self.full_endo(), cal.KLj0),
            'pKLj':Variable(self.full_endo(), cal.pKLj0),
            'Yj':Variable(self.full_endo(), cal.Yj0),
            'pYj':Variable(self.full_endo(), cal.pYj0),
            'Cj': Variable(self.full_endo(), cal.Cj0),
            'pCj': Variable(self.full_endo(), cal.pCj0),
            'Mj': Variable(self.idx_1D(exo_names = ["ENERGY"]), cal.Mj0),
            'pMj': Variable(self.idx_1D(exo_names = ["ENERGY"]), cal.pMj0),
            'Xj': Variable(self.idx_1D(exo_names = ["ENERGY"]), cal.Xj0),
            'Dj': Variable(self.full_endo(), cal.Dj0),
            'pDj': Variable(self.full_endo(), cal.pDj0),
            'Sj': Variable(self.full_endo(), cal.Sj0),
            'pSj': Variable(self.full_endo(), cal.pSj0),
            'Gj': Variable(self.full_endo(),cal.Gj0),
            'Ij': Variable(self.full_endo(), cal.Ij0),
            #'Yij': Variable(self.idx_2D(exo_names=[("PRIMARY","SECONDARY")]), cal.Yij0),
            'Yij': Variable(self.full_endo(), cal.Yij0),
            'I':Variable(self.full_endo(), cal.I0),
            #'w':Variable(self.full_endo(), cal.w),            
            'aYij':Variable(self.idx_2D(endo_names = endo_aYij_indexes), cal.aYij),

            'pY_Ej':Variable(self.full_endo(), cal.pY_Ej),
           #'E_P':Variable(self.full_endo(), cal.E_P),
           # 'E_T':Variable(self.full_endo(), cal.E_T),
           # 'E_B':Variable(self.full_endo(), cal.E_B),

            #'lambda_E':Variable(self.full_endo(), cal.lambda_E),
            #'lambda_nE':Variable(self.full_endo(), cal.lambda_nE),

            'pE_TT':Variable(self.full_endo(), cal.pE_TT),
            'pE_TnT':Variable(self.full_endo(), cal.pE_TnT),
            'pE_B':Variable(self.full_endo(), cal.pE_B),
            'pE_Pj':Variable(self.full_endo(), cal.pE_Pj),
            'pE_Ej':Variable(self.idx_1D(endo_names = ["ENERGY"]), cal.pE_Ej),

            'wG':Variable(self.full_exo(), cal.wG),
            'GDPreal':Variable(self.full_exo(), cal.GDPreal),
            #'tauL':Variable(self.full_exo(), cal.tauL0),
            'tauSj':Variable(self.full_exo(), cal.tauSj0),
            'tauYj':Variable(self.full_exo(), cal.tauYj0),
            #'pXj':Variable(self.idx_1D(endo_names = ["ENERGY"]), cal.pXj),
            'bKLj':Variable(self.full_exo(), cal.bKLj),
            'pCtp':Variable(self.full_exo(), cal.pCjtp),
            'Ctp':Variable(self.full_exo(), cal.Ctp),
            'Gtp':Variable(self.full_exo(), cal.Gtp),
            'Itp':Variable(self.full_exo(), cal.Itp),
            'pXtp':Variable(self.full_exo(), cal.pXtp),
            'Xtp':Variable(self.full_exo(), cal.Xtp),
            'Mtp':Variable(self.full_exo(), cal.Mtp),
            'alphaKj':Variable(self.full_exo(), cal.alphaKj),
            'alphaLj':Variable(self.full_exo(), cal.alphaLj),
            'aKLj':Variable(self.idx_1D(endo_names = ["ENERGY"]), cal.aKLj),
            'alphaCj0':Variable(self.full_exo(), cal.alphaCj0),
            'alphaGj':Variable(self.full_exo(), cal.alphaGj),
            'alphaIj':Variable(self.full_exo(), cal.alphaIj),
            
            #'alphaXj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.alphaXj),
            #'alphaDj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.alphaDj),
            #'betaDj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.betaDj),
            #'betaMj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.betaMj),
            
            'alphaXj':Variable(self.full_exo(), cal.alphaXj0),
            'alphaDj':Variable(self.full_exo(), cal.alphaDj0),
            'betaDj':Variable(self.full_exo(), cal.betaDj0),
            'betaMj':Variable(self.full_exo(), cal.betaMj0),
            
            'thetaj':Variable(self.full_exo(), cal.thetaj),
            'csij':Variable(self.full_exo(), cal.csij),
            'sigmaXj':Variable(self.full_exo(), cal.sigmaXj),
            'sigmaSj':Variable(self.full_exo(), cal.sigmaSj),
            'sigmaKLj':Variable(self.full_exo(), cal.sigmaKLj),
            #'aYij':Variable(self.idx_2D(endo_names=[("PRIMARY","SECONDARY")]), cal.aYij),
            'delta':Variable(self.full_exo(), cal.delta),

            'YE_Pj':Variable(self.full_exo(), cal.YE_Pj),
            'YE_Tj':Variable(self.full_exo(), cal.YE_Tj),
            'YE_Bj':Variable(self.full_exo(), cal.YE_Bj),
            'C_EB':Variable(self.full_exo(), cal.C_EB),
            'C_ET':Variable(self.full_exo(), cal.C_ET),

            'YE_Ej':Variable(self.full_exo(), cal.YE_Ej),

            'aKLj0':Variable(self.full_exo(), cal.aKLj0),
            'aYij0':Variable(self.full_exo(), cal.aYij0),
            'pXj':Variable(self.idx_1D(exo_names = ["ENERGY","SERVICES"]), cal.pXj),

            'rhoB':Variable(self.full_exo(), cal.rhoB),
            'rhoTT':Variable(self.full_exo(), cal.rhoTT),
            'rhoTnT':Variable(self.full_exo(), cal.rhoTnT),
            'rhoPj':Variable(self.full_exo(), cal.rhoPj),

            'lambda_KLM':Variable(self.full_exo(), cal.lambda_KLM),
            #'lambda_XMj':Variable(self.full_exo(), cal.lambda_XMj),
            
            "R_E":Variable(self.full_endo(), cal.R_E),
            "R_nE":Variable(self.full_endo(), cal.R_nE),

            'betaCj_nE':Variable(self.full_exo(), cal.betaCj_nE),
            'u_C':Variable(self.full_endo(), cal.u_C),
            "gammaCj_nE":Variable(self.full_exo(), cal.gammaCj_nE),
            "A_Cj_nE":Variable(self.full_exo(), cal.A_Cj_nE),
            "normalisation_factor":Variable(self.full_exo(), cal.normalisation_factor),
            
            #'lambda_XMj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.lambda_XMj),
            
            }

        Knext = Variable(self.full_endo(), cal.K0u_next) if closure in ["neokeynesian1","neokeynesian2"] else Variable(self.full_endo(), cal.K0next)

        self.commonDict = {**self.commonDict, **{'Knext': Knext}} if endoKnext else self.commonDict

        self.closure=closure

        self.variables_dict = self.assignClosure(cal)

        self.endogeouns_dict = self.to_endo_dict()

        self.exogenous_dict = self.to_exo_dict()

###########   BOUNDS    ################


bounds={
    'Rreal':(0,np.inf),
    'CPI':(-np.inf,np.inf),
    'pL':(0,np.inf),
    'pK':(0,np.inf),
    'pI':(0,np.inf),
    'w':(0,np.inf),
    'K':(0,np.inf),
    'L':(0,np.inf),
    'B':(-np.inf,np.inf),
    'wB':(-1,1),
    'GDPPI':(-np.inf,np.inf),
    'GDPreal':(-np.inf,np.inf), 
    'GDP':(-np.inf,np.inf),
    'R':(0,np.inf),
    'I':(0,np.inf),
    'Ri':(0,np.inf),
    'Rg':(0,np.inf),
    'l':(0,np.inf),
    'bKL':(-np.inf,np.inf),
    'tauL':(-1,1),
    'tauSj':(-1,1),
    'tauYj':(-1,1),
    'T':(-np.inf,np.inf),
    'w_real':(0,np.inf),
    'uL':(0,1),
    'sigmaw':(-np.inf,np.inf),
    'pK_real':(0,np.inf),
    'alphapK':(0,np.inf),
    'sigmapK':(-np.inf,np.inf),
    'uK':(0,1),
    'Knext':(0,np.inf),
    
    'bKLj':(0,np.inf),    
    'Cj':(0,np.inf),
    'pCj':(0,np.inf),
    'Sj':(0,np.inf),
    'pSj':(0,np.inf),
    'Kj':(0,np.inf),
    'Lj':(0,np.inf),
    'KLj':(0,np.inf),
    'pKLj':(0,np.inf),
    'Dj':(0,np.inf),
    'pDj':(0,np.inf),
    'Xj':(0,np.inf),
    'pXj':(0,np.inf),
    'Yj':(0,np.inf),
    'pYj':(0,np.inf),
    'Mj':(0,np.inf),
    'pMj':(0,np.inf),  
    'Gj':(0,np.inf),
    'Ij':(0,np.inf),
    'pCtp':(0,np.inf),
    'Ctp':(0,np.inf),
    'Gtp':(0,np.inf),
    'Itp':(0,np.inf),
    'pXtp':(0,np.inf),
    'Xtp':(0,np.inf),
    'Mtp':(0,np.inf),
    'alphaKj':(0,1),
    'alphaLj':(0,1),
    'aKLj':(0,np.inf),
    'alphaCj':(0,1),
    'alphaXj':(0,np.inf),
    'alphaDj':(0,np.inf),
    'alphaGj':(0,np.inf),
    'alphaIj':(0,np.inf),    
    'alphalj':(0,1),  
    'alphaw':(0,np.inf),  
    'alphaIK':(0,np.inf),  
    'betaDj':(0,np.inf),
    'betaMj':(0,np.inf),
    'sigmaXj':(-np.inf,np.inf),
    'sigmaSj':(-np.inf,np.inf),
    'sigmaKLj':(-np.inf,np.inf),    
    'wG':(0,1),
    'wI':(0,1),
    'sD':(0,1),
    'sK':(0,1),
    'sL':(0,1),
    'sG':(0,1),
    'aYij':(0,np.inf),
    'Yij':(0,np.inf),
    'delta':(0,1),
    'thetaj':(-np.inf,np.inf),
    'csij':(-np.inf,np.inf),
    
    
    'alphaCj0':(0,np.inf),
    'YE_Pj':(0,np.inf),
    'YE_Tj':(0,np.inf),
    'C_EB':(0,np.inf),
    'YE_Bj':(0,np.inf),
    'C_ET':(0,np.inf),
    
    'pE_TT':(0,np.inf),
    'pE_TnT':(0,np.inf),
    'pE_B':(0,np.inf),
    'pE_Pj':(0,np.inf),
    'pY_Ej':(0,np.inf),
    'E_P':(0,np.inf),
    'E_T':(0,np.inf),
    'E_B':(0,np.inf),
    'tau_Ej':(-1,1),
    'lambda_E':(0,np.inf),
    'lambda_nE':(0,np.inf),
    'pE_Ej':(0,np.inf),
    'YE_Ej':(0,np.inf),
    'lambda_KLM':(0,np.inf),
    'aKLj0':(0,np.inf),
    'aYij0':(0,np.inf),
    'pXj0':(0,np.inf),
    'rhoB':(0,np.inf),
    'rhoTT':(0,np.inf),
    'rhoTnT':(0,np.inf),
    'rhoPj':(0,np.inf),
    'lambda_XMj':(0,1),
    
    
    'alphaXj0':(0,np.inf),
    'alphaDj0':(0,np.inf),
    'betaDj0':(0,np.inf),
    'betaMj0':(0,np.inf),
    "R_E":(0,np.inf),
    "R_nE":(0,np.inf),
    
    'betaCj_nE': (-np.inf,np.inf),
    'u_C':(-np.inf,np.inf),
    "gammaCj_nE":(-np.inf,np.inf),
    "A_Cj_nE":(0,np.inf),
    "normalisation_factor":(-np.inf,np.inf),

        }

