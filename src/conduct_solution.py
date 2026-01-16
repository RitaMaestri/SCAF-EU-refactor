import numpy as np
import copy
import sys

def get_value_at_position(position, dictionary):
    key, pos = position
    if pos is None:  # Scalar value
        return dictionary[key]
    elif len(pos) == 1:  # Array with one index
        return dictionary[key][pos]
    elif len(pos) == 2:  # Array with two indices
        i, j = pos
        return dictionary[key][i][j]
    else:
        raise ValueError("Invalid position format")


def set_value_at_position(dictionary, position, value):
    new_dictionary = copy.deepcopy(dictionary)

    key, pos = position

    if pos is None:  # Scalar value
        new_dictionary[key] = value
    elif len(pos) == 1:  # Array with one index
        idx = pos[0]
        new_dictionary[key][idx] = value
    elif len(pos) == 2:  # Array of array with two indices
        i, j = pos
        new_dictionary[key][i,j] = value
    else:
        raise ValueError("Invalid position format")

    return new_dictionary


def dictionary_gr(dictionary_t, dictionary_t0):
    gr = {}
    for key in dictionary_t.keys():
        if key in dictionary_t0.keys():
            gr[key] = dictionary_t[key] / dictionary_t0[key] - 1
    return gr


def get_all_positions(dictionary):
    all_positions = []

    for key, value in dictionary.items():
        if not isinstance(value, np.ndarray):  # Scalar value
            all_positions.append((key, None))
        elif isinstance(value, np.ndarray):  # Array value
            for idx, elem in np.ndenumerate(value):
                all_positions.append((key, list(idx)))
        elif isinstance(value, (list, tuple)):  # Array of array value
            for i, sub_array in enumerate(value):
                for j, sub_elem in enumerate(sub_array):
                    all_positions.append((key, [i, j]))

    return all_positions
    


def find_positions_above_threshold(dictionary, threshold):
    positions_above_threshold = []
    all_positions = get_all_positions(dictionary)

    for position in all_positions:
        value = get_value_at_position(position, dictionary)
        if abs(value) > threshold:
            positions_above_threshold.append(position)

    return positions_above_threshold



        
def get_values_at_positions(positions, target_dictionary):
    values = []
    for position in positions:
        values.append(get_value_at_position(position, target_dictionary))
    return values


def step_forward_value(old_value, new_value, growth_rate):
    step=abs(growth_rate*old_value)

    
    if not np.sign(old_value) ==  np.sign(new_value):
        print("change in sign of the parameters not implemented yet")
        sys.exit()
    
    if old_value == new_value or (old_value - step <= new_value <= old_value + step) :
        return new_value
    elif old_value - step > new_value :
        return old_value - step
    elif old_value + step < new_value :
        return old_value + step




def step_forward(position, previous_dictionary, target_dictionary, growth_rate):
    old_value = get_value_at_position(position, previous_dictionary)
    target_value = get_value_at_position(position, target_dictionary)
    updated_value=step_forward_value(old_value,target_value, growth_rate)
    
    updated_dictionary=set_value_at_position(previous_dictionary, position, updated_value)
    return updated_dictionary



def set_initial_parameters(previous_dictionary, parameters_target, positions):
    initial_parameters = parameters_target
    for position in positions:
        old_value = get_value_at_position(position, previous_dictionary)
        initial_parameters = set_value_at_position(initial_parameters, position, old_value)
        
    return initial_parameters



def smooth_par_evolution(parameters_previous_iteration, parameters_target, growth_rate, positions):
    parameters_current_iteration=set_initial_parameters(parameters_previous_iteration, parameters_target, positions)

    for position in positions:
        parameters_current_iteration = step_forward(position, parameters_current_iteration, parameters_target, growth_rate)

    return parameters_current_iteration


def are_dicts_equal(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False  # Different keys

    for key in dict1.keys():
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.allclose(val1, val2, equal_nan=True):
                return False
            
        elif isinstance(val1, float) and np.isnan(val1):
            if not np.isnan(val2):
                return False
            
        elif val1 != val2:
            return False

    return True




def conduct_solution(parameters_origin, parameters_target, system, bounds_variables, N, threshold= 0.07, growth_rate=0.01):
    
    # parameters_origin = {'pL': 1.15, 'pK': np.array([0.9, 2.02, 4.02]), 'B': np.array([[3.01, 4.01], [5.5, 2.02]]),'C': np.array([np.nan, 2.02, 4.01])}
    # parameters_target = {'pL': 1, 'pK': np.array([1, 2, 4]), 'B': np.array([[3, 4.001], [5, 2]]), 'C': np.array([np.nan, 2, 4])}
    # dictionary_grr=dictionary_gr(parameters_origin,parameters_target)
    # positions=find_positions_above_threshold(dictionary_grr, threshold=0.1)
    
    
    parameters_gr=dictionary_gr( parameters_target,parameters_origin  )
    positions=find_positions_above_threshold(parameters_gr, threshold)
    
    
    variables=System.df_to_dict(var=True, t=years[t-1])
    
    parameters=parameters_origin
    
    while not are_dicts_equal(parameters, parameters_target) :

        parameters = smooth_par_evolution(parameters, parameters_target, growth_rate, positions)

        solution = dict_least_squares( system, variables, parameters, bounds_variables, N, verb=1, check=True)
        
        variables = solution.dvar
        maxerror=max(abs( system(solution.dvar, parameters)))
        if maxerror>1e-06:
            print("the system doesn't converge, maxerror=",maxerror)
            sys.exit()
    
    return variables