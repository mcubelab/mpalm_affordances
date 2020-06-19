import sympy
import numpy as np

def get_symbols(data):
    symbols_set = sympy.Matrix(data).free_symbols
    symbol_list = []
    while True:
        try:
            symbol_list.append(symbols_set.pop())
        except:
            break
    return symbol_list



def symbolic_matrix(size, matrix_name):
    symbolic_list = []
    for i in range(size[0]):
        symbolic_list_tmp = []
        for j in range(size[1]):
            symbolic_list_tmp.append(symbols(matrix_name + str(i) + str(j)))
        symbolic_list.append(symbolic_list_tmp)
    return np.array(symbolic_list)

def symbolic_jacobian(A, x_var):
    A_matrix = sympy.Matrix(A)
    x_var_matrix = sympy.Matrix(x_var)
    try:
        return np.array(A_matrix.jacobian(x_var_matrix)).astype(float)
    except:
        return np.array(A_matrix.jacobian(x_var_matrix))

def symbolic_multiply(mat1, mat2):
    result_list = []
    is_vector = False
    if len(mat2.shape)==1:
        mat2 = mat2.reshape((mat2.shape[0], 1))
        is_vector = True

    for i in range(mat1.shape[0]):
        result_list_tmp = []
        # iterate through columns of Y
        for j in range(mat2.shape[1]):
            # iterate through rows of Y
            result = 0
            for k in range(mat2.shape[0]):
                result += mat1[i][k] * mat2[k][j]
            result_list_tmp.append(result)
        result_list.append(result_list_tmp)
    if is_vector:
        return np.ndarray.flatten(np.array(result_list));
    else:
        return np.array(result_list)


def symbolic_substitute(matrix, variable_list, value_list):
    symbolic_list = []
    if len(matrix.shape)==1:
        for i in range(matrix.shape[0]):
            new_expr = matrix[i]
            if isinstance(new_expr, tuple(sympy.core.all_classes)):
                for k in range(len(variable_list)):
                    new_expr = new_expr.subs(variable_list[k], value_list[k])
            symbolic_list.append(new_expr)
    else:
        for i in range(matrix.shape[0]):
            symbolic_list_tmp = []
            for j in range(matrix.shape[1]):
                new_expr = matrix[i][j]
                if isinstance(new_expr, tuple(sympy.core.all_classes)):
                    for k in range(len(variable_list)):
                        new_expr = new_expr.subs(variable_list[k], value_list[k])
                symbolic_list_tmp.append(new_expr)
            symbolic_list.append(symbolic_list_tmp)
    try:
        return np.array(symbolic_list).astype(np.float)
    except:
        return np.array(symbolic_list)

