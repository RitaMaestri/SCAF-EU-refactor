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

    def check_calibration_value_bounds(self):
        cal = self.calibration_value
        lb, ub = self.bounds
        if isinstance(cal, np.ndarray):
            if np.any(cal < lb) or np.any(cal > ub):
                violating = np.where((cal < lb) | (cal > ub))
                raise ValueError(
                    f"Variable '{self.name}': calibration value has entries outside bounds [{lb}, {ub}]. "
                    f"Violating indices: {violating}, values: {cal[violating]}"
                )
        else:
            if cal < lb or cal > ub:
                raise ValueError(
                    f"Variable '{self.name}': calibration value {cal} is outside bounds [{lb}, {ub}]"
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

        self.check_calibration_value_bounds()

        self.shape = self.assign_shape()
        
        self.exo_mask = self.assign_exo_mask(status, endo_names, exo_names)
        
        if self.exo_mask is None:
            raise RuntimeError("exo mask not assigned for variable "+ name)
        
        self.endo_mask = ~ self.exo_mask