#SOLVEURS ALTERNATIVES

from scipy import optimize
import numpy as np
from math import sqrt
import sys

# TYPE CONVERTERS 


def to_array(dvar):  #takes dict returns array
    var=[np.array(dvar[k]) for k in dvar.keys()]
    return np.array(np.hstack([var[i].flatten() for i in range(len(var))]))

def non_zero_index(dvar):
    array_var= to_array(dvar)
    return np.where(array_var != 0)[0]

def to_dict(vec, dvec, is_variable, variables_specs=None):  #takes array WITHOUT ZEROS, returns dict of arrays (of equal dimensions and keys as dvec)
    lengths = np.array([int(np.prod(np.shape(item))) for item in dvec.values()])
    keys = list(dvec.keys())
    #add zeros to vector
    if(is_variable):
        zeros=np.zeros(len(to_array(dvec)))
        zeros[non_zero_index(dvec)]=vec
        vec=zeros
    #create array of arrays
    chunks = np.split(vec,np.cumsum(lengths))[:-1]
    #reshape each chunk using idx_labels from variables_specs (supports non-square matrices)
    vec = []
    for key, chunk in zip(keys, chunks):
        spec = variables_specs.get(key) if variables_specs is not None else None
        if (spec is not None
                and getattr(spec, 'dimension', None) == 'matrix'
                and len(getattr(spec, 'idx_labels', [])) == 2):
            shape = (len(spec.idx_labels[0]), len(spec.idx_labels[1]))
            chunk = np.reshape(chunk, shape)
        vec.append(chunk)
    for i in range(len(vec)):
        if isinstance(vec[i], np.ndarray) and vec[i].size == 1:
            vec[i]=vec[i].item()
    return dict(zip(keys, vec))


def same_number(var,system):
    len_var=len(var)
    len_sys=len(system)
    if len_var != len_sys:
        raise ValueError(f"system has {len_sys} equations but there are {len_var} variables")

########################  FSOLVE  ########################
def dict_fsolve(f, dvar, dpar,N):
    print("I'm in fsolve, before solving")
    result = optimize.fsolve(
        lambda x,y: f(to_dict(x,dvar), y, N), # wrap the argument in a dict
        to_array(dvar), # unwrap the initial dictionary
        args= dpar
    )
    print("I'm in fsolve, after solving")
    result.dvar= to_dict(result, dvar,N)
    result.d= {**result.dvar, **dpar}
    return result;


########################  MINIMIZE  ########################
def cost_function(array):
    return np.sum(array**2)

def convert_bounds_to_minimize(lsq_bounds):
    """
    Convert bounds from scipy.optimize.least_squares format to scipy.optimize.minimize format.

    Parameters:
    - lsq_bounds: tuple of (lower_bounds, upper_bounds)
        - lower_bounds: array-like, specifies lower bounds for each variable
        - upper_bounds: array-like, specifies upper bounds for each variable

    Returns:
    - minimize_bounds: list of tuples, [(lower_bound, upper_bound), ...]
    """
    lower_bounds, upper_bounds = lsq_bounds
    
    # Ensure bounds are numpy arrays
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    
    # Convert to minimize format
    minimize_bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
    
    return minimize_bounds

def dict_minimize(f, dvar, dpar,N, bounds, constraint):
    result = optimize.minimize(
        fun=lambda x,y: cost_function(f(to_dict(x,dvar,True), to_dict(y,dpar,False))),# wrap the argument in a dict
        x0=to_array(dvar), # unwrap the initial dictionary
        bounds=convert_bounds_to_minimize(bounds),
        args= to_array(dpar),
        method='trust-constr', 
        constraints=constraint,
        options={"maxiter":60000,'gtol': 1e-14, 'disp': True}
    )
    result.dvar= to_dict(result.x, dvar,True)
    result.d= {**result.dvar, **dpar}
    return result;


####################  LEAST_SQUARE  #########################

def dict_least_squares(f, dvar, dpar, bounds, variables_specs, verb=1, check=True):
    #check same number
    if check:
        non_zero_dvar=to_array(dvar)[to_array(dvar)!=0]
        same_number(non_zero_dvar,f(dvar,dpar))
    
    result = optimize.least_squares(
        lambda x,y: f(to_dict(x,dvar,is_variable=True,variables_specs=variables_specs), to_dict(y,dpar,is_variable=False,variables_specs=variables_specs)),# wrap the argument in a dict
        to_array(dvar)[to_array(dvar)!=0], # unwrap the initial dictionary
        bounds=bounds,
        args= list([to_array(dpar)],),
        verbose=verb
    )

    result.dvar= to_dict(result.x, dvar, is_variable=True, variables_specs=variables_specs)
    result.d= {**result.dvar, **dpar}
    return result;

##########   BASINHOPPING   ###############

class MyBounds:
    def __init__(self, bounds ):
        self.xmin,self.xmax= [[row[i] for row in bounds] for i in (0,1)]
        self.bounds = [np.array([row[0]+1e-14,row[1]]) for row in bounds]

        self.epsilon = 1e-14
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x < self.xmax))
        tmin = bool(np.all(x > self.xmin))
        return tmax and tmin


def dict_basinhopping(f, dvar, dpar, mybounds,N):
    result = optimize.basinhopping(
        lambda x,y: cost_function(f(to_dict(x,dvar,is_variable=True), to_dict(y,dpar,is_variable=False))),# wrap the argument in a dict
        x0=to_array(dvar), # unwrap the initial dictionary     
        #niter=2,
        minimizer_kwargs = dict(method="L-BFGS-B",  bounds=mybounds.bounds, args= to_array(dpar)),
        accept_test=mybounds,
        stepsize=10
    )
    result.dvar= to_dict(result.x, dvar, is_variable=True)
    result.d= {**result.dvar, **dpar}
    return result


##########   SHGO NOT WORKING   ###############


def dict_shgo(f, dvar, dpar,bounds): # unwrap the initial dictionary
    result = optimize.shgo(
        lambda x,y: cost_function(f(to_dict(x,dvar), to_dict(y,dpar))),# wrap the argument in a dict
        bounds=bounds,
        args= list([to_array(dpar)],),
        minimizer_kwargs = dict(method="L-BFGS-B"),
    )
    result.dvar= to_dict(result.x, dvar)
    result.d= {**result.dvar, **dpar}
    return result;

