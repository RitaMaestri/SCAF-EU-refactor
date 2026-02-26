import numpy as np



def compare_dictionaries(dict1, dict2):
    if dict1.keys() != dict2.keys():
        print("The dictionaries have different keys")
    else:
        equal_keys = True
        for key in dict1.keys():
            value1 = dict1[key]
            value2 = dict2[key]
            if type(value1) != type(value2):
                equal_keys = False
                print(f"The key '{key}' has different value types")
            elif hasattr(value1, "__len__"):
                if not np.array_equal(value1, value2):
                    equal_keys = False
                    if len(value1) != len(value2):
                        print(f"The key '{key}' has arrays with different lengths")
                    else:
                        unequal_indexes = np.where(value1 != value2)[0]
                        print(f"The key '{key}' has unequal values at indexes: {unequal_indexes}")
                        print(f"Value 1 at unequal indexes: {value1[unequal_indexes]}")
                        print(f"Value 2 at unequal indexes: {value2[unequal_indexes]}")
            else:
                if value1 != value2:
                    equal_keys = False
                    print(f"The key '{key}' has unequal float values")
        if equal_keys:
            print("The dictionaries are equal")


def count_elements(dictionary):
    count = 0
    for value in dictionary.values():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                count += len(value)
            elif value.ndim == 2:
                count += value.size
        elif isinstance(value, (float, int)):
            count += 1
        else:
            print("Unsupported type:", type(value))
    return count



def filter_nan_values(original_dict):
    new_dict = {}
    
    for key, value in original_dict.items():
        if isinstance(value, np.ndarray):
            if not np.any(np.isnan(value)):
                new_dict[key] = value
        elif np.isnan(value):
            continue
        else:
            new_dict[key] = value
    
    return new_dict

def compare_dictionaries(dict1, dict2):
    unequal_keys = []

    for key in dict1.keys():
        if key in dict2.keys():
            value1 = dict1[key]
            value2 = dict2[key]

            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                if not np.array_equal(value1, value2):
                    unequal_keys.append(key)
            elif value1 != value2:
                unequal_keys.append(key)
        else:
            unequal_keys.append(key)

    # Check for keys that exist in dict2 but not in dict1
    for key in dict2.keys():
        if key not in dict1.keys():
            unequal_keys.append(key)

    return unequal_keys


def compare_dictionaries_keys(dictionary1, dictionary2):
    # Find keys present only in dictionary1
    keys_only_in_dictionary1 = set(dictionary1.keys()) - set(dictionary2.keys())

    # Find keys present only in dictionary2
    keys_only_in_dictionary2 = set(dictionary2.keys()) - set(dictionary1.keys())

    return keys_only_in_dictionary1, keys_only_in_dictionary2


def find_keys_with_large_elements(dictionary, threshold=10e-2):
    keys_with_large_elements = []

    for key, value in dictionary.items():
        # Check if at least one element in the array is greater than one
        if isinstance(value, (list, np.ndarray)):
            if np.any(abs(np.array(value)) > threshold):
                keys_with_large_elements.append(key)

    return keys_with_large_elements

def find_keys_with_negative_elements(dictionary, threshold=10e-2):
    keys_with_large_elements = []

    for key, value in dictionary.items():
        # Check if at least one element in the array is greater than one
        if isinstance(value, (list, np.ndarray)):
            if np.any(np.array(value) < threshold):
                keys_with_large_elements.append(key)

    return keys_with_large_elements

def column(matrix, i):
    return [row[i] for row in matrix]

