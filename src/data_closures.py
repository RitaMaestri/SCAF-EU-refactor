import numpy as np 
from simple_calibration import calibrationVariables
from import_GTAP_data import sectors
from itertools import product



#######  Class Variable  #############

class Variable():

    def full_endo(self):
        #everything is False
        return np.zeros(self.shape, dtype=bool)
    

    def full_exo(self):
        #everything is True
        return np.ones(self.shape, dtype=bool)


    def idx_1D(self, exo_names=None, endo_names=None):
        if (not exo_names == None) and endo_names==None:
            indices_list = [index for index, value in enumerate(self.idx_labels) if value in exo_names]
        elif (not endo_names==None) and exo_names == None:
            indices_list = [index for index, value in enumerate(self.idx_labels) if value not in endo_names]
        else:
            raise ValueError("wrong arguments for idx_1D")

        msk = np.zeros(len(self.idx_labels), dtype=bool)
        msk[indices_list] = True
        
        return msk


    #takes couple of sectors names: first name is the row identifier, second name is the column identifier. expected [(sec1,sec2),(sec1,sec3),(sec2,sec4)]   
    def idx_2D(self, exo_names=None, endo_names=None):
        
        if (not exo_names == None) and endo_names==None:
            indexes_list = [(self.idx_labels[0].index(row), self.idx_labels[1].index(col)) for row, col in exo_names]
            # Sort the indexes_list based on the first element (index_a) and then the second element (index_b)
            sorted_indexes_list = sorted(indexes_list, key=lambda x: (x[0], x[1]))
            rows,cols=zip(*sorted_indexes_list)

        elif (not endo_names==None) and exo_names == None:
            indexes_list = [(self.idx_labels[0].index(row), self.idx_labels[1].index(col)) for row, col in endo_names]
            #matrix_size = (len(self.idx_labels[0]), len(self.idx_labels[1]))
            all_indexes_set = set(product(range(len(self.idx_labels[0])), range(len(self.idx_labels[1]))))
            # Create a set of present indexes (row_index, column_index) pairs
            present_indexes_set = set(indexes_list)
            # Find the complementary set of indexes (not present in the matrix)
            complementary_indexes_set = all_indexes_set.difference(present_indexes_set)
            # Sort the complementary indexes to maintain order
            sorted_indexes_list = sorted(list(complementary_indexes_set))
        else:
            raise ValueError("wrong arguments for idx_2D")
        
        rows,cols=zip(*sorted_indexes_list)
        idx_positions= [list(rows), list(cols)]

        msk = np.zeros((len(self.idx_labels[0]),len(self.idx_labels[1])), dtype=bool)
        msk[idx_positions] = True

        return msk


    def assign_shape(self):
        if self.idx_labels is not None:
            if self.dimension == "vector":
                shape = (len(self.idx_labels),)
            elif self.dimension == "matrix":
                shape = (len(self.idx_labels[0]), len(self.idx_labels[1]))
            elif self.dimension == "scalar":   
                shape = (1,)
            else:
                raise ValueError("wrong dimension specification for variable "+ self.name)
        return shape
            

    def assign_exo_mask(self,status, endo_names, exo_names):
        if status is not None:
            if status == "endo":
                return self.full_endo()
            if status == "exo":
                return self.full_exo()
            else:
                raise ValueError("wrong status specification for variable "+ self.name)

        if endo_names is not None:
            if self.dimension == "vector":
                return self.idx_1D(endo_names = endo_names)
            elif self.dimension == "matrix":
                return self.idx_2D(endo_names = endo_names)
        if exo_names is not None:
            if self.dimension == "vector":
                return self.idx_1D(exo_names = exo_names)
            elif self.dimension == "matrix":
                return self.idx_2D(exo_names = exo_names)

    def assign_calibration_value(self, calibration_value):
        if isinstance(calibration_value, np.ndarray) and calibration_value.size == 1:
            return calibration_value.item()
        else:
            return calibration_value

    def check_dimension_idx_labels(self):
        if self.dimension == "scalar":
            if self.idx_labels != []:
                raise ValueError(
                    f"Variable '{self.name}': dimension is 'scalar' but idx_labels is not empty: {self.idx_labels}"
                )
        elif self.dimension == "vector":
            if not (isinstance(self.idx_labels, list) and len(self.idx_labels) > 0
                    and not isinstance(self.idx_labels[0], list)):
                raise ValueError(
                    f"Variable '{self.name}': dimension is 'vector' but idx_labels is not a non-empty flat list: {self.idx_labels}"
                )
        elif self.dimension == "matrix":
            if not (isinstance(self.idx_labels, list) and len(self.idx_labels) == 2
                    and isinstance(self.idx_labels[0], list) and isinstance(self.idx_labels[1], list)):
                raise ValueError(
                    f"Variable '{self.name}': dimension is 'matrix' but idx_labels is not a list of 2 lists: {self.idx_labels}"
                )
        else:
            raise ValueError(f"Variable '{self.name}': unknown dimension '{self.dimension}'")

    def check_calibration_value_dimension(self):
        cal = self.calibration_value
        if self.dimension == "scalar":
            if isinstance(cal, np.ndarray) and cal.ndim > 0:
                raise ValueError(
                    f"Variable '{self.name}': dimension is 'scalar' but calibration value is an array with shape {cal.shape}"
                )
        elif self.dimension == "vector":
            expected_len = len(self.idx_labels)
            if not isinstance(cal, np.ndarray) or cal.ndim != 1 or len(cal) != expected_len:
                actual = cal.shape if isinstance(cal, np.ndarray) else type(cal).__name__
                raise ValueError(
                    f"Variable '{self.name}': dimension is 'vector' with {expected_len} labels, "
                    f"but calibration value has shape/type {actual}"
                )
        elif self.dimension == "matrix":
            expected_shape = (len(self.idx_labels[0]), len(self.idx_labels[1]))
            if not isinstance(cal, np.ndarray) or cal.shape != expected_shape:
                actual = cal.shape if isinstance(cal, np.ndarray) else type(cal).__name__
                raise ValueError(
                    f"Variable '{self.name}': dimension is 'matrix' with expected shape {expected_shape}, "
                    f"but calibration value has shape/type {actual}"
                )

    def __init__(self, name, calibration_value, dimension, idx_labels, is_t_minus_one, bounds, status=None, endo_names=None, exo_names=None):
        
        self.name = name

        self.dimension = dimension

        self.idx_labels = idx_labels

        self.is_t_minus_one = is_t_minus_one

        self.bounds = bounds

        self.calibration_value = self.assign_calibration_value(calibration_value)

        self.check_dimension_idx_labels()

        self.check_calibration_value_dimension()

        self.shape = self.assign_shape()
        
        self.exo_mask = self.assign_exo_mask(status, endo_names, exo_names)
        
        if self.exo_mask is None:
            raise RuntimeError("exo mask not assigned for variable "+ name)
        
        self.endo_mask = ~ self.exo_mask




endo_aYij_indexes = [("ENERGY",x) for x in sectors]
sectors_nE = [x for x in sectors if x != "ENERGY"]
endo_aYij_indexes.extend([(x,"ENERGY") for x in sectors_nE])

cal = calibrationVariables()

VARIABLES_SPECS = {
    'K':Variable(name= "K",
                calibration_value=cal.K0, 
                dimension='scalar', 
                idx_labels=[], 
                is_t_minus_one=False, 
                bounds=(0, np.inf), 
                status="exo"),

    'wI':Variable(name= "wI", 
                  calibration_value=cal.wI, 
                  dimension='scalar', 
                  idx_labels=[], 
                  is_t_minus_one=False, 
                  bounds=(0, 1), 
                  status="exo"),

    'wB':Variable(name= "wB", 
                  calibration_value=cal.wB, 
                  dimension='scalar', 
                  idx_labels=[], 
                  is_t_minus_one=False, 
                  bounds=(-1, 1), 
                  status="exo"),

    'L':Variable(name= "L", 
                 calibration_value=cal.L0, 
                 dimension='scalar', 
                 idx_labels=[], 
                 is_t_minus_one=False, 
                 bounds=(0, np.inf), 
                 status="exo"),

    'pL': Variable(name= "pL",
                 calibration_value=cal.pL0,
                 dimension='scalar',
                 idx_labels=[],
                 is_t_minus_one=False,
                 bounds=(0,np.inf),
                 status="endo"),

    'pK':Variable(name= "pK",
				  calibration_value=cal.pK0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  status="endo"),

    'pI':Variable(name= "pI",
				  calibration_value=cal.pI0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  status="endo"),

    'B':Variable(name= "B",
				  calibration_value=cal.B0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="endo"),

    'R':Variable(name= "R",
				  calibration_value=cal.R0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  status="endo"),

    'Ri':Variable(name= "Ri",
				  calibration_value=cal.Ri0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  status="endo"),

    'Rg':Variable(name= "Rg",
				  calibration_value=cal.Rg0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  status="endo"),

    'bKL':Variable(name= "bKL",
				  calibration_value=cal.bKL,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="endo"),

    'GDPPI':Variable(name= "GDPPI",
				  calibration_value=cal.GDPPI,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="endo"),

    'GDP':Variable(name= "GDP",
				  calibration_value=cal.GDPreal,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="endo"),
                  
    'Kj':Variable(name="Kj",
				  calibration_value=cal.Kj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  status="endo"),

    'Lj':Variable(name="Lj",
				  calibration_value=cal.Lj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'KLj':Variable(name="KLj",
				  calibration_value=cal.KLj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pKLj':Variable(name="pKLj",
				  calibration_value=cal.pKLj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Yj':Variable(name="Yj",
				  calibration_value=cal.Yj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pYj':Variable(name="pYj",
				  calibration_value=cal.pYj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Cj': Variable(name="Cj",
				  calibration_value=cal.Cj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pCj': Variable(name="pCj",
				  calibration_value=cal.pCj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Mj': Variable(name="Mj",
				  calibration_value=cal.Mj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  exo_names=["ENERGY"]),

    'pMj': Variable(name="pMj",
				  calibration_value=cal.pMj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  exo_names=["ENERGY"]),

    'Xj': Variable(name="Xj",
				  calibration_value=cal.Xj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  exo_names=["ENERGY"]),

    'Dj': Variable(name="Dj",
				  calibration_value=cal.Dj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pDj': Variable(name="pDj",
				  calibration_value=cal.pDj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Sj': Variable(name="Sj",
				  calibration_value=cal.Sj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pSj': Variable(name="pSj",
				  calibration_value=cal.pSj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Gj': Variable(name="Gj",
				  calibration_value=cal.Gj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Ij': Variable(name="Ij",
				  calibration_value=cal.Ij0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'Yij': Variable(name="Yij",
				  calibration_value=cal.Yij0,
				  dimension='matrix',
				  idx_labels=[sectors,
				  sectors],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'I':Variable(name="I",
				  calibration_value=cal.I0,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'aYij':Variable(name="aYij",
				  calibration_value=cal.aYij0 ,
				  dimension='matrix',
				  idx_labels=[sectors,
				  sectors],
				  is_t_minus_one=False,
				  bounds=(0,np.inf),
				  endo_names=endo_aYij_indexes),

    'pY_Ej':Variable(name="pY_Ej",
				  calibration_value=cal.pY_Ej,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pE_TT':Variable(name="pE_TT",
				  calibration_value=cal.pE_TT,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pE_TnT':Variable(name="pE_TnT",
				  calibration_value=cal.pE_TnT,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pE_B':Variable(name="pE_B",
				  calibration_value=cal.pE_B,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'pE_Pj':Variable(name="pE_Pj",
				  calibration_value=cal.pE_Pj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),


    'pE_Ej':Variable(name="pE_Ej",
				  calibration_value=cal.pE_Ej,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  endo_names=["ENERGY"]),

    'wG':Variable(name="wG",
				  calibration_value=cal.wG,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, 1),
				  status="exo"),

    'GDPreal':Variable(name="GDPreal",
				  calibration_value=cal.GDPreal,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'tauSj':Variable(name="tauSj",
				  calibration_value=cal.tauSj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-1,
				  1),
				  status="exo"),

    'tauYj':Variable(name="tauYj",
				  calibration_value=cal.tauYj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-1,
				  1),
				  status="exo"),

    'bKLj':Variable(name="bKLj",
				  calibration_value=cal.bKLj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'pCtp':Variable(name="pCtp",
				  calibration_value=cal.pCjtp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),

    'Ctp':Variable(name="Ctp",
				  calibration_value=cal.Ctp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),

    'Gtp':Variable(name="Gtp",
				  calibration_value=cal.Gtp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),

    'Itp':Variable(name="Itp",
				  calibration_value=cal.Itp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),

    'pXtp':Variable(name="pXtp",
				  calibration_value=cal.pXtp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),

    'Xtp':Variable(name="Xtp",
				  calibration_value=cal.Xtp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),

    'Mtp':Variable(name="Mtp",
				  calibration_value=cal.Mtp,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=True,
				  bounds=(0, np.inf),
				  status="exo"),
                  
    'alphaKj':Variable(name="alphaKj",
				  calibration_value=cal.alphaKj ,
				  dimension='vector',
				  idx_labels=sectors ,
				  is_t_minus_one=False ,
				  bounds=(0,1),
				  status = "exo"),

    'alphaLj':Variable(name="alphaLj",
				  calibration_value=cal.alphaLj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, 1),
				  status="exo"),

    'aKLj':Variable(name="aKLj",
				  calibration_value=cal.aKLj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  endo_names=["ENERGY"]),

    'alphaCj0':Variable(name="alphaCj0",
				  calibration_value=cal.alphaCj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'alphaGj':Variable(name="alphaGj",
				  calibration_value=cal.alphaGj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'alphaIj':Variable(name="alphaIj",
				  calibration_value=cal.alphaIj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'alphaXj':Variable(name="alphaXj",
				  calibration_value=cal.alphaXj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'alphaDj':Variable(name="alphaDj",
				  calibration_value=cal.alphaDj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'betaDj':Variable(name="betaDj",
				  calibration_value=cal.betaDj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'betaMj':Variable(name="betaMj",
				  calibration_value=cal.betaMj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'thetaj':Variable(name="thetaj",
				  calibration_value=cal.thetaj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'csij':Variable(name="csij",
				  calibration_value=cal.csij,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'sigmaXj':Variable(name="sigmaXj",
				  calibration_value=cal.sigmaXj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'sigmaSj':Variable(name="sigmaSj",
				  calibration_value=cal.sigmaSj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'sigmaKLj':Variable(name="sigmaKLj",
				  calibration_value=cal.sigmaKLj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'delta':Variable(name="delta",
				  calibration_value=cal.delta,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, 1),
				  status="exo"),

    'YE_Pj':Variable(name="YE_Pj",
				  calibration_value=cal.YE_Pj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'YE_Tj':Variable(name="YE_Tj",
				  calibration_value=cal.YE_Tj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'YE_Bj':Variable(name="YE_Bj",
				  calibration_value=cal.YE_Bj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'C_EB':Variable(name="C_EB",
				  calibration_value=cal.C_EB,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'C_ET':Variable(name="C_ET",
				  calibration_value=cal.C_ET,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'YE_Ej':Variable(name="YE_Ej",
				  calibration_value=cal.YE_Ej,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'aYE_Bj':Variable(name="aYE_Bj",
				  calibration_value=cal.aYE_Bj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'aYE_Pj':Variable(name="aYE_Pj",
				  calibration_value=cal.aYE_Pj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'aYE_Tj':Variable(name="aYE_Tj",
				  calibration_value=cal.aYE_Tj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'aKLj0':Variable(name="aKLj0",
				  calibration_value=cal.aKLj0,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'aYij0':Variable(name="aYij0",
				  calibration_value=cal.aYij0,
				  dimension='matrix',
				  idx_labels=[sectors,
				  sectors],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'pXj':Variable(name="pXj",
				  calibration_value=cal.pXj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  exo_names=["ENERGY","SERVICES"]),

    'rhoB':Variable(name="rhoB",
				  calibration_value=cal.rhoB,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'rhoTT':Variable(name="rhoTT",
				  calibration_value=cal.rhoTT,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf), 
                  status="exo"),

    'rhoTnT':Variable(name="rhoTnT",
				  calibration_value=cal.rhoTnT,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'rhoPj':Variable(name="rhoPj",
				  calibration_value=cal.rhoPj,
				  dimension='vector',
				  idx_labels=sectors,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    'lambda_KLM':Variable("lambda_KLM",
				  cal.lambda_KLM,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    "R_E":Variable("R_E",
				  cal.R_E,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),
                  
    "R_nE":Variable("R_nE",
				  cal.R_nE,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="endo"),

    'betaCj_nE':Variable(name="betaCj_nE",
				  calibration_value=cal.betaCj_nE,
				  dimension='vector',
				  idx_labels=sectors_nE,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    'u_C':Variable(name="u_C",
				  calibration_value=cal.u_C,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="endo"),

    "gammaCj_nE":Variable(name="gammaCj_nE",
				  calibration_value=cal.gammaCj_nE,
				  dimension='vector',
				  idx_labels=sectors_nE,
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    "A_Cj_nE":Variable(name="A_Cj_nE",
				  calibration_value=cal.A_Cj_nE,
				  dimension='vector',
				  idx_labels=sectors_nE,
				  is_t_minus_one=False,
				  bounds=(0, np.inf),
				  status="exo"),

    "normalisation_factor":Variable(name="normalisation_factor",
				  calibration_value=cal.normalisation_factor,
				  dimension='scalar',
				  idx_labels=[],
				  is_t_minus_one=False,
				  bounds=(-np.inf,np.inf),
				  status="exo"),

    }


    #'lambda_XMj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.lambda_XMj),
    #'lambda_XMj':Variable(self.full_exo(), cal.lambda_XMj),
    #'aYij':Variable(self.idx_2D(endo_names=[("PRIMARY","SECONDARY")]), cal.aYij),
    #'alphaXj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.alphaXj),
    #'alphaDj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.alphaDj),
    #'betaDj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.betaDj),
    #'betaMj':Variable(self.idx_1D(endo_names = ["STEEL","CHEMICAL"]), cal.betaMj),
    #'pXj':Variable(idx_1D(endo_names = ["ENERGY"]), cal.pXj),
    #'tauL':Variable(self.full_exo(), cal.tauL0),
    #'E_P':Variable(self.full_endo(), cal.E_P),
    # 'E_T':Variable(self.full_endo(), cal.E_T),
    # 'E_B':Variable(self.full_endo(), cal.E_B),
    #'w':Variable(self.full_endo(), cal.w),
    #'Yij': Variable(self.idx_2D(exo_names=[("PRIMARY","SECONDARY")]), cal.Yij0),
    #'T':Variable(self.full_endo(), cal.T0),
    #'lambda_KLM':Variable(self.idx_1D(endo_names = ["ENERGY"]), cal.lambda_KLM),
    #'CPI':Variable(self.full_endo(), cal.CPI),
    #'Rreal':Variable(self.full_endo(), cal.R0),
    #'lambda_E':Variable(self.full_endo(), cal.lambda_E),
    #'lambda_nE':Variable(self.full_endo(), cal.lambda_nE),


########################################
###########   BOUNDS    ################
########################################


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
    'aYE_Bj':(0,np.inf),
    'aYE_Pj':(0,np.inf),
    'aYE_Tj':(0,np.inf),
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



class endo_exo_indexes:
    def full_endo(self):
        return []
        
    def full_exo(self):
        return slice(None)
    
    def idx_1D(self, sectors_names, exo_names=None, endo_names=None):
        if (not exo_names == None) and endo_names==None:
            indices_list = [index for index, value in enumerate(sectors_names) if value in exo_names]
        elif (not endo_names==None) and exo_names == None:
            indices_list = [index for index, value in enumerate(sectors_names) if value not in endo_names]
        else:
            raise ValueError("wrong arguments for idx_1D")
        return indices_list



    #takes couple of sectors names: first name is the row identifier, second name is the column identifier. expected [(sec1,sec2),(sec1,sec3),(sec2,sec4)]   
    def idx_2D(sectors_names, exo_names=None, endo_names=None):
        
        if (not exo_names == None) and endo_names==None:
            indexes_list = [(sectors_names.index(row), sectors_names.index(col)) for row, col in exo_names]
            # Sort the indexes_list based on the first element (index_a) and then the second element (index_b)
            sorted_indexes_list = sorted(indexes_list, key=lambda x: (x[0], x[1]))
            rows,cols=zip(*sorted_indexes_list)

        elif (not endo_names==None) and exo_names == None:
            indexes_list = [(sectors_names.index(row), sectors_names.index(col)) for row, col in endo_names]
            matrix_size = len(sectors_names)
            all_indexes_set = set(product(range(matrix_size), repeat=2))
            # Create a set of present indexes (row_index, column_index) pairs
            present_indexes_set = set(indexes_list)
            # Find the complementary set of indexes (not present in the matrix)
            complementary_indexes_set = all_indexes_set.difference(present_indexes_set)
            # Sort the complementary indexes to maintain order
            sorted_indexes_list = sorted(list(complementary_indexes_set))
        else:
            raise ValueError("wrong arguments for idx_2D")
        
        rows,cols=zip(*sorted_indexes_list)
        return [list(rows), list(cols)]

class calibrationDict(endo_exo_indexes):

    def to_endo_dict(self):
        result_dict = {}
        for key, variable in self.variables_dict.items():
            if isinstance(variable.calibration_value, np.ndarray):
                masked_array = variable.calibration_value[variable.endo_mask]
                if masked_array.size > 0:
                    result_dict[key] = masked_array
            elif variable.endo_mask:
                result_dict[key]=variable.calibration_value  
        
        return result_dict
    
    
    def to_exo_dict(self):
        result_dict = {}
        for key, variable in self.variables_dict.items():
            result_dict[key]=variable.calibration_value
            if isinstance(variable.calibration_value, np.ndarray):
                result_dict[key][variable.endo_mask]=float("nan")  
            elif variable.endo_mask:
                    result_dict[key] = float("nan")  
        return result_dict
    
    def __init__(self, variables_dict):

        self.variables_dict = variables_dict

        self.endogeouns_dict = self.to_endo_dict()

        self.exogenous_dict = self.to_exo_dict()



""" def assignClosure(self,cal):
        
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
                            .py
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
            raise ValueError("this closure doesn't exist") """