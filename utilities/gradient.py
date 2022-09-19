import numpy as np
import sys
from typing import Callable



def central_difference(obj_fun: Callable, x, dim_num):
    """
    Central-difference estimation of a gradient

    Params
    ----------
    obj_fun : function
        objective function
    x : numpy.longdouble or numpy.ndarray
        point at which gradient is estimated
    dim_num : int
        number of dimensions

    Return
    ----------
    resvals : numpy.ndarray
        estimated gradient at point x
    """

    if sys.platform == "win32":
        epsilon=np.longdouble(1.e-5)
    elif sys.platform == "linux":
        epsilon=np.longdouble(1.e-6)
    else:
        raise SystemError("Unfortunately, your platform is not supported yet. Supported platforms are Windows and Linux")
    resvals = np.zeros(dim_num, dtype=np.longdouble)
    if dim_num==1:
        resvals[0] = (obj_fun(x+epsilon)-obj_fun(x-epsilon))/(2*epsilon)
    else:
        for dim in range(dim_num):
            e_i = np.zeros(dim_num, dtype=np.longdouble)
            e_i[dim] = 1
            resvals[dim] = (obj_fun(x+epsilon*e_i)-obj_fun(x-epsilon*e_i))/(2*epsilon)
    return resvals 

def forward_difference(obj_fun: Callable, x, dim_num):
    """
    Forward-difference estimation of a gradient

    Params
    ----------
    obj_fun : function
        objective function
    x : numpy.longdouble or numpy.ndarray
        point at which gradient is estimated
    dim_num : int
        number of dimensions

    Return
    ----------
    resvals : numpy.ndarray
        estimated gradient at point x
    """

    if sys.platform == "win32":
        epsilon=np.longdouble(1.e-7)
    elif sys.platform == "linux":
        epsilon=np.longdouble(1.e-9)
    else:
        raise SystemError("Unfortunately, your platform is not supported yet. Supported platforms are Windows and Linux")
    resvals = np.empty(dim_num, dtype=np.longdouble)
    if dim_num==1:
        resvals[0] = (obj_fun(x+epsilon)-obj_fun(x))/epsilon
    else:
        for dim in range(dim_num):
            e_i = np.zeros(dim_num, dtype=np.longdouble)
            e_i[dim] = 1
            resvals[dim] = (obj_fun(x+epsilon*e_i)-obj_fun(x))/epsilon
    return resvals 

def gradient(obj_fun: Callable, x, method="central"):
    """
    Estimate gradient of the objective function at x

    Params
    ----------
    obj_fun : function
        objective function
    x : numpy.longdouble or numpy.ndarray
        point at which the gradient is estimated
    method : string
        method used for the gradient estimation;
        supported methods are "central" and "forward"

    Return
    ----------
    resvals : numpy.longdouble or numpy.ndarray
        estimated gradient at point x
    """

    if isinstance(x,np.longdouble):
        dim_num = 1
    elif isinstance(x, np.ndarray):
        if len(x.shape) != 1:
            raise ValueError("x should have only 1 dimension, got {}".format(len(x.shape)))
        if isinstance(x[0],np.longdouble):
            dim_num = len(x)
        else:
            raise TypeError("Elements of x should be of type np.longdouble, got {}".format(type(x[0])))
    else:
        raise TypeError("x should be of type numpy.longdouble or numpy.ndarray, got {}".format(type(x)))
    
    estimation_functions = [central_difference, forward_difference] 
    if method == "central":
        est_fun = estimation_functions[0]
    elif method == "forward":
        est_fun = estimation_functions[1]
    else:
        raise ValueError("Expected method is central or forward, got {}".format(method))

    resvals = est_fun(obj_fun, x, dim_num)
    if dim_num==1:
        return resvals[0]
    else:
        return resvals

def gradient_ls(obj_fun, x, a, b):
    """
    Estimate gradient for least-squares problem at x

    Params
    ----------
    obj_fun : function
        objective function
    x : numpy.ndarray
        point at which the gradient is estimated
    a : numpy.ndarray
        generated argument points
    b : numpy.ndarray
        function values corresponding to a

    Return
    ----------
    res : numpy.ndarray
        estimated gradient at point x
    """

    m = len(a)
    pol_degree = len(x)-1
    res = np.zeros(pol_degree+1, dtype=np.longdouble)
    for j in range(m):
        c = [a[j]**n for n in range(pol_degree+1)]
        res += np.dot(c@x-b[j], c)
    return res
