# -*- coding: utf-8 -*-
import math



# transpose of a martix
def transpose(mat):
    dim = (len(mat), len(mat[0]))
    res = [[0] * dim[0] for i in range(dim[1])]
    for i in range(dim[1]):
        for j in range(dim[0]):
            res[i][j] = mat[j][i]
    return res


# inner product between two matrix or a matrix and a vector
def dot(mat, mat_vct):
    if type_of(mat_vct) == 'matrix':  # mat_vct is a matrix
        if len(mat[0]) != len(mat_vct):
            raise Exception("Incoordinate Matrix Dimensions")
        else:
            res = [[0] * len(mat_vct[0]) for i in range(len(mat))]
            for i in range(len(mat)):
                for j in range(len(mat_vct[0])):
                    for k in range(len(mat_vct)):
                        res[i][j] = res[i][j] + mat[i][k] * mat_vct[k][j]
            return res
    else:  # mat_vct is a vector
        if len(mat[0]) != len(mat_vct):
            raise Exception("Incoordinate Matrix and Vector Dimensions")
        else:
            res = [0] * len(mat)
            for i in range(len(mat)):
                for j in range(len(mat_vct)):
                    res[i] = res[i] + mat[i][j] * mat_vct[j]
            return res


# get type of x: matrix, vector, scalar
def type_of(x):
    if type(x) == list:
        if type(x[0]) == list:
            return 'matrix'
        else:
            return 'vector'
    else:
        return 'scalar'


# addition function realizing:
# element-wise addition:    matrix + matrix, vector + vector
# broadcasting addition:    matrix + vector, vector + matrix
#                           matrix + scalar, scalar + matrix
#                           vector + scalar, scalar + vector
def add(x, y):
    type_x = type_of(x)
    type_y = type_of(y)
    x_is_mat = type_x == 'matrix'
    x_is_vct = type_x == 'vector'
    x_is_scl = type_x == 'scalar'
    y_is_mat = type_y == 'matrix'
    y_is_vct = type_y == 'vector'
    y_is_scl = type_y == 'scalar'
    # case1: matrix + matrix
    if x_is_mat and y_is_mat:
        if len(x) == len(y) and len(x[0]) == len(y[0]):
            res = [[0] * len(x[0]) for i in range(len(x))]
            for i in range(len(x)):
                for j in range(len(x[0])):
                    res[i][j] = x[i][j] + y[i][j]
            return res
        else:
            raise Exception("Incoordinate Matrix Dimensions")
    # case2: vector + vector
    if x_is_vct and y_is_vct:
        if len(x) == len(y):
            res = [0] * len(x)
            for i in range(len(x)):
                res[i] = x[i] + y[i]
            return res
        else:
            raise Exception("Incoordinate Vector Dimensions")
    # case3: matrix + vector
    if x_is_mat and y_is_vct:
        if len(x) == len(y):
            res = [[0] * len(x[0]) for i in range(len(x))]
            for i in range(len(x)):
                for j in range(len(x[0])):
                    res[i][j] = x[i][j] + y[i]
            return res
        else:
            raise Exception("Incoordinate Matrix and Vector Dimensions")
    # case4: vector + matrix
    if x_is_vct and y_is_mat:
        if len(x) == len(y):
            res = [[0] * len(y[0]) for i in range(len(y))]
            for i in range(len(y)):
                for j in range(len(y[0])):
                    res[i][j] = y[i][j] + x[i]
            return res
        else:
            raise Exception("Incoordinate Matrix and Vector Dimensions")
    # case5: matrix + scalar
    if x_is_mat and y_is_scl:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = x[i][j] + y
        return res
    # case6: scalar + matrix
    if x_is_scl and y_is_mat:
        res = [[0] * len(y[0]) for i in range(len(y))]
        for i in range(len(y)):
            for j in range(len(y[0])):
                res[i][j] = y[i][j] + x
        return res
    # case7: vector + scalar
    if x_is_vct and y_is_scl:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = x[i] + y
        return res
    # case8: scalar + vector
    if x_is_scl and y_is_vct:
        res = [0] * len(y)
        for i in range(len(y)):
            res[i] = y[i] + x
        return res


# substraction function realizing:
# element-wise addition:    matrix - matrix, vector - vector
# broadcasting addition:    matrix - vector, vector - matrix
#                           matrix - scalar, scalar - matrix
#                           vector - scalar, scalar - vector
def sub(x, y):
    type_x = type_of(x)
    type_y = type_of(y)
    x_is_mat = type_x == 'matrix'
    x_is_vct = type_x == 'vector'
    x_is_scl = type_x == 'scalar'
    y_is_mat = type_y == 'matrix'
    y_is_vct = type_y == 'vector'
    y_is_scl = type_y == 'scalar'
    # case1: matrix - matrix
    if x_is_mat and y_is_mat:
        if len(x) == len(y) and len(x[0]) == len(y[0]):
            res = [[0] * len(x[0]) for i in range(len(x))]
            for i in range(len(x)):
                for j in range(len(x[0])):
                    res[i][j] = x[i][j] - y[i][j]
            return res
        else:
            raise Exception("Incoordinate Matrix Dimensions")
    # case2: vector - vector
    if x_is_vct and y_is_vct:
        if len(x) == len(y):
            res = [0] * len(x)
            for i in range(len(x)):
                res[i] = x[i] - y[i]
            return res
        else:
            raise Exception("Incoordinate Vector Dimensions")
    # case3: matrix - vector
    if x_is_mat and y_is_vct:
        if len(x) == len(y):
            res = [[0] * len(x[0]) for i in range(len(x))]
            for i in range(len(x)):
                for j in range(len(x[0])):
                    res[i][j] = x[i][j] - y[i]
            return res
        else:
            raise Exception("Incoordinate Matrix and Vector Dimensions")
    # case4: vector - matrix
    if x_is_vct and y_is_mat:
        if len(x) == len(y):
            res = [[0] * len(y[0]) for i in range(len(y))]
            for i in range(len(y)):
                for j in range(len(y[0])):
                    res[i][j] = x[i] - y[i][j]
            return res
        else:
            raise Exception("Incoordinate Matrix and Vector Dimensions")
    # case5: matrix - scalar
    if x_is_mat and y_is_scl:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = x[i][j] - y
        return res
    # case6: scalar - matrix
    if x_is_scl and y_is_mat:
        res = [[0] * len(y[0]) for i in range(len(y))]
        for i in range(len(y)):
            for j in range(len(y[0])):
                res[i][j] = x - y[i][j]
        return res
    # case7: vector - scalar
    if x_is_vct and y_is_scl:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = x[i] - y
        return res
    # case8: scalar - vector
    if x_is_scl and y_is_vct:
        res = [0] * len(y)
        for i in range(len(y)):
            res[i] = x - y[i]
        return res


# uni-dimensional sigmoid function
def sigmoid_dim1(x):
    return 1 / (1 + math.exp(-x))


# element-wise sigmoid
def sigmoid(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = sigmoid_dim1(x[i][j])
        return res
    if x_is_vct:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = sigmoid_dim1(x[i])
        return res
    if x_is_scl:
        return sigmoid_dim1(x)


# element-wise tanh
def tanh(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = math.tanh(x[i][j])
        return res
    if x_is_vct:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = math.tanh(x[i])
        return res
    if x_is_scl:
        return math.tanh(x)


# element-wise product
def times(x, y):
    type_x = type_of(x)
    type_y = type_of(y)
    x_is_mat = type_x == 'matrix'
    x_is_vct = type_x == 'vector'
    x_is_scl = type_x == 'scalar'
    y_is_mat = type_y == 'matrix'
    y_is_vct = type_y == 'vector'
    y_is_scl = type_y == 'scalar'
    # matrix-matrix element-wise product
    if x_is_mat and y_is_mat:
        if len(x) == len(y) and len(x[0]) == len(y[0]):
            res = [[0] * len(x[0]) for i in range(len(x))]
            for i in range(len(x)):
                for j in range(len(x[0])):
                    res[i][j] = x[i][j] * y[i][j]
            return res
        else:
            raise Exception("Incoordinate Matrix Dimensions")
    # vector-vector element-wise product
    if x_is_vct and y_is_vct:
        if len(x) == len(y):
            res = [0] * len(x)
            for i in range(len(x)):
                res[i] = x[i] * y[i]
            return res
        else:
            raise Exception("Incoordinate Vector Dimensions")
    # scalar times vector
    if x_is_scl and y_is_vct:
        res = [0] * len(y)
        for i in range(len(y)):
            res[i] = x * y[i]
        return res
    # vector times scalar
    if x_is_vct and y_is_scl:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = y * x[i]
        return res
    # scalar times matrix
    if x_is_scl and y_is_mat:
        res = [[0] * len(y[0]) for i in range(len(y))]
        for i in range(len(y)):
            for j in range(len(y[0])):
                res[i][j] = x * y[i][j]
        return res
    # matrix times scalar
    if x_is_mat and y_is_scl:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = y * x[i][j]
        return res


# outer product between two vectors
def cross(u, v):
    u_is_vct = type_of(u) == 'vector'
    v_is_vct = type_of(v) == 'vector'
    if u_is_vct and v_is_vct:
        res = [[0] * len(v) for i in range(len(u))]
        for i in range(len(u)):
            for j in range(len(v)):
                res[i][j] = u[i] * v[j]
        return res
    else:
        raise Exception("Outer Product Requires Two Vectors")


# inverse
def inverse(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = float(1.0 / x[i][j])
        return res
    if x_is_vct:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = float(1.0 / x[i])
        return res
    if x_is_scl:
        return float(1.0 / x)


# square root
def sq_root(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = math.sqrt(float(x[i][j]))
        return res
    if x_is_vct:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = math.sqrt(float(x[i]))
        return res
    if x_is_scl:
        return math.sqrt(float(x))


# Gaussian kernel
def gaussian_kernel(label, miu, sigma):
    if len(label) != len(miu):
        raise Exception("Incoordinate Label and Miu Dimensions")
    else:
        return math.exp((-1) * norm2_sq(sub(label, miu)) / 2 / sigma / sigma) / \
               math.pow(sigma, len(miu)) / \
               math.pow(math.sqrt(2 * math.pi), len(miu))


# Norm2 square
def norm2_sq(vct):
    norm = 0
    for i in range(len(vct)):
        norm = norm + vct[i] * vct[i]
    return norm


# Mean value
def mean(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [0] * len(x[0])
        for i in range(len(x[0])):
            for j in range(len(x)):
                res[i] = res[i] + x[j][i]
            res[i] = float(res[i]) / len(x)
        return res
    if x_is_vct:
        res = 0
        for i in range(len(x)):
            res = res + x[i]
        return float(res) / len(x)
    if x_is_scl:
        return x


# Standard Deviation
def std(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    mean_x = mean(x)
    if x_is_mat:
        res = [0] * len(x[0])
        for i in range(len(x[0])):
            for j in range(len(x)):
                res[i] = res[i] + (x[j][i] - mean_x[i]) * (x[j][i] - mean_x[i])
            res[i] = math.sqrt(float(res[i]) / len(x))
        return res
    if x_is_vct:
        res = 0
        for i in range(len(x)):
            res = res + (x[i] - mean_x) * (x[i] - mean_x)
        return math.sqrt(float(res) / len(x))
    if x_is_scl:
        return 0


# Gaussian normalize
def normalize_gauss(x, mean_x, std_x):
    x_is_mat = type_of(x) == 'matrix'
    mean_is_vct = type_of(mean_x) == 'vector'
    std_is_vct = type_of(std_x) == 'vector'
    if x_is_mat and mean_is_vct and std_is_vct:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                if std_x[j] != 0:
                    res[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
                else:
                    res[i][j] = x[i][j] - mean_x[j]
        return res
    else:
        raise Exception("Gaussian Normalize Function Requires Feature Matrix; \
                        Mean and Standard Deviation Vector")


# Gaussian denormalize
def denormalize_gauss(x, mean_x, std_x):
    x_is_mat = type_of(x) == 'matrix'
    mean_is_vct = type_of(mean_x) == 'vector'
    std_is_vct = type_of(std_x) == 'vector'
    if x_is_mat and mean_is_vct and std_is_vct:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = x[i][j] * std_x[j] + mean_x[j]
        return res
    else:
        raise Exception("Gaussian Denormalize Function Requires Normalized Feature Matrix;\
                        Mean and Standard Deviation Vector")


# uniform normalization
def normalize_uniform(x, max_x, min_x):
    x_is_mat = type_of(x) == 'matrix'
    max_is_vct = type_of(max_x) == 'vector'
    min_is_vct = type_of(min_x) == 'vector'
    if x_is_mat and max_is_vct and min_is_vct:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                u = (max_x[j] + min_x[j]) / 2.0
                v = (max_x[j] - min_x[j]) / 2.0
                if v != 0:
                    res[i][j] = (x[i][j] - u) / v
                else:
                    res[i][j] = x[i][j] - u
        return res
    else:
        raise Exception("Uniform Normalize Function Requires Feature Matrix; \
                        Max and Min Vector")


# uniform denormalization
def denormalize_uniform(xn, max_x, min_x):
    x_is_mat = type_of(xn) == 'matrix'
    max_is_vct = type_of(max_x) == 'vector'
    min_is_vct = type_of(min_x) == 'vector'
    if x_is_mat and max_is_vct and min_is_vct:
        res = [[0] * len(xn[0]) for i in range(len(xn))]
        for i in range(len(xn)):
            for j in range(len(xn[0])):
                u = (max_x[j] + min_x[j]) / 2.0
                v = (max_x[j] - min_x[j]) / 2.0
                res[i][j] = v * xn[i][j] + u
        return res
    else:
        raise Exception("Uniform Denormalize Function Requires Feature Matrix; \
                        Max and Min Vector")


# element-wise ceil
def ceil(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = int(math.ceil(x[i][j]))
        return res
    if x_is_vct:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = int(math.ceil(x[i]))
        return res
    if x_is_scl:
        return int(math.ceil(x))


# element-wise trunc
def trunc(x):
    x_is_mat = type_of(x) == 'matrix'
    x_is_vct = type_of(x) == 'vector'
    x_is_scl = type_of(x) == 'scalar'
    if x_is_mat:
        res = [[0] * len(x[0]) for i in range(len(x))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[i][j] = int(trunc_ew(x[i][j]))
        return res
    if x_is_vct:
        res = [0] * len(x)
        for i in range(len(x)):
            res[i] = int(trunc_ew(x[i]))
        return res
    if x_is_scl:
        return int(trunc_ew(x))


def trunc_ew(x):
    int_part = math.ceil(x)
    first_decimal = math.floor((x - int_part) * 10)
    if first_decimal < 5:
        return math.floor(x)
    else:
        return math.ceil(x)


# count
def count(x):
    x_is_mat = type_of(x) == 'matrix'
    if x_is_mat:
        res = [0] * len(x[0])
        for i in range(len(x)):
            for j in range(len(x[0])):
                res[j] = res[j] + x[i][j]
        return res
    else:
        raise Exception("Input Requires Matrix")

def count_list(x):
    x_is_mat = type_of(x) == 'matrix'
    if x_is_mat:
        res = []
        for j in range(len(x[0])):
            temp=[]
            for i in range(len(x)):
                temp.append(x[i][j])
            res.append(temp)
        return res
    else:
        raise Exception("Input Requires Matrix")

def sqr_avg_square_sum(x):
    avg_ss = 0.0
    for i in range(len(x)):
        avg_ss += x[i] ** 2
    avg_ss /= len(x)
    avg_ss = math.sqrt(avg_ss)
    return avg_ss


def find_max(x):
    x_is_mat = type_of(x) == 'matrix'
    if x_is_mat:
        res = [0] * len(x[0])
        for j in range(len(x[0])):
            max_xj = -1000
            for i in range(len(x)):
                if x[i][j] > max_xj:
                    max_xj = x[i][j]
            res[j] = max_xj
    return res


def find_min(x):
    x_is_mat = type_of(x) == 'matrix'
    if x_is_mat:
        res = [0] * len(x[0])
        for j in range(len(x[0])):
            min_xj = 1000
            for i in range(len(x)):
                if x[i][j] < min_xj:
                    min_xj = x[i][j]
            res[j] = min_xj
    return res
