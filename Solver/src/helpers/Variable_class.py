import numpy as np 
from itertools import product



#######  Class Variable  #############

class Variable():

    def full_endo(self):
        """Return an exogeneity mask with all entries set to endogenous.

        Returns:
            Boolean array matching ``self.shape`` with all values set to False.
        """
        # In exo_mask convention, False means endogenous.
        return np.zeros(self.shape, dtype=bool)
    

    def full_exo(self):
        """Return an exogeneity mask with all entries set to exogenous.

        Returns:
            Boolean array matching ``self.shape`` with all values set to True.
        """
        # In exo_mask convention, True means exogenous.
        return np.ones(self.shape, dtype=bool)


    def idx_1D(self, exo_names=None, endo_names=None):
        """Build a 1D exogeneity mask from selected labels.

        Exactly one between ``exo_names`` and ``endo_names`` must be provided.

        Args:
            exo_names: Labels to be marked exogenous.
            endo_names: Labels to be marked endogenous.

        Returns:
            1D boolean mask aligned with ``self.idx_labels``.
        """
        if (not exo_names == None) and endo_names==None:
            indices_list = [index for index, value in enumerate(self.idx_labels) if value in exo_names]
        elif (not endo_names==None) and exo_names == None:
            indices_list = [index for index, value in enumerate(self.idx_labels) if value not in endo_names]
        else:
            raise ValueError("wrong arguments for idx_1D")

        msk = np.zeros(len(self.idx_labels), dtype=bool)
        msk[indices_list] = True
        
        return msk


    def idx_2D(self, exo_names=None, endo_names=None):
        """Build a 2D exogeneity mask from row/column label pairs.

        Each pair is interpreted as ``(row_label, col_label)``.
        Exactly one between ``exo_names`` and ``endo_names`` must be provided.

        Args:
            exo_names: Pairs to be marked exogenous.
            endo_names: Pairs to be marked endogenous.

        Returns:
            2D boolean mask with the same shape as the variable matrix.
        """
        
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
        """Infer the numerical shape from variable dimension and labels.

        Returns:
            Tuple describing the variable shape.
        """
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
        """Assign the exogeneity mask from either status or endo_names or exo_names. 
        The three arguments are mutually exclusive, and at least one must be provided.

        Args:
            status: if provided, must be either "endo" or "exo" and it sets the status of all elements in the variable.
            endo_names: Labels/pairs declared endogenous. The remaining will be exogenous.
            exo_names: Labels/pairs declared exogenous. The remaining will be endogenous.

        Returns:
            Boolean mask where True = exogenous and False = endogenous.
        """
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
        """Normalize scalar calibration values represented as length-1 arrays.

        Args:
            calibration_value: Raw calibration value from specs.

        Returns:
            Scalar value for 1-element arrays, otherwise unchanged input.
        """
        if isinstance(calibration_value, np.ndarray) and calibration_value.size == 1:
            return calibration_value.item()
        else:
            return calibration_value


    def check_dimension_idx_labels(self):
        """Validate consistency between dimension and ``idx_labels`` structure.

        Raises:
            ValueError: If labels do not match the declared dimension.
        """
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
        """Validate calibration value shape against the declared dimension.

        Raises:
            ValueError: If calibration value shape/type is inconsistent.
        """
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
        """Validate calibration values against variable bounds.

        Raises:
            ValueError: If any scalar/array element is outside bounds.
        """
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

    def initialize_calibration_value(self, cal, mapping):
        """Assign and validate the calibration value from a calibrationVariables instance.

        Args:
            cal: A calibrationVariables instance.
            mapping: Dict mapping variable names to calibrationVariables attribute names.
        """
        attr = mapping[self.name]
        self.calibration_value = self.assign_calibration_value(getattr(cal, attr))
        self.check_calibration_value_dimension()
        self.check_calibration_value_bounds()


    def __init__(self, name, dimension, idx_labels, is_t_minus_one, bounds, status=None, endo_names=None, exo_names=None):
        """Initialize a Variable and compute all derived masks/metadata.

        Calibration values are not set here. Call ``initialize_calibration_value``
        after construction to assign and validate them.

        Args:
            name: Variable name.
            dimension: One of ``scalar``, ``vector``, or ``matrix``.
            idx_labels: Labels used for vector/matrix dimensions.
            is_t_minus_one: False or linked variable name for lag logic.
            bounds: Tuple of lower and upper numerical bounds.
            status: Optional global exogeneity status (``endo``/``exo``).
            endo_names: Optional label selection for endogenous entries.
            exo_names: Optional label selection for exogenous entries.
        """
        
        self.name = name

        self.dimension = dimension

        self.idx_labels = idx_labels

        self.is_t_minus_one = is_t_minus_one

        self.bounds = bounds

        self.calibration_value = None

        self.check_dimension_idx_labels()

        self.shape = self.assign_shape()
        
        self.exo_mask = self.assign_exo_mask(status, endo_names, exo_names)
        
        if self.exo_mask is None:
            raise RuntimeError("exo mask not assigned for variable " + name)
        
        self.endo_mask = ~ self.exo_mask