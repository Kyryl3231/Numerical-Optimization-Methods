import numpy as np
import sys
from typing import Callable
from gradient import gradient


def hessian(obj_fun: Callable, x):
    """
    Estimate Hessian matrix of the objective function at x

    Params
    ----------
    obj_fun : function
        objective function
    x : numpy.longdouble or numpy.ndarray
        point at which Hessian is estimated
    
    Return
    ----------
    resvals : numpy.longdouble or numpy.ndarray
        estimated Hessian at point x
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
    
    if sys.platform == "win32":
        epsilon=np.longdouble(1.e-5)
    elif sys.platform == "linux":
        epsilon=np.longdouble(1.e-6)
    else:
        raise SystemError("Unfortunately, your platform is not supported yet. Supported platforms are Windows and Linux")
    
    resvals = np.empty(shape=(dim_num,dim_num), dtype=np.longdouble)
    
    if dim_num==1:
       resvals[0,0] = (gradient(obj_fun, x+epsilon) - gradient(obj_fun,x))/epsilon 
    else: 
        for i in range(dim_num):
            e_i = np.zeros(dim_num, dtype=np.longdouble)
            e_i[i] = 1
            for j in range(i,dim_num):
                e_j = np.zeros(dim_num, dtype=np.longdouble)
                e_j[j] = 1
                resvals[i,j] = (obj_fun(x+epsilon*e_i+epsilon*e_j) - obj_fun(x+epsilon*e_i) - obj_fun(x+epsilon*e_j) + obj_fun(x)) / epsilon**2
                resvals[j,i] = resvals[i,j]
    
    if dim_num==1:
        return resvals[0,0]
    else:
        return resvals


def hessian_ls(x, a):
    """
    Estimate Hessian for least-squares problem at x

    Params
    ----------
    x : numpy.ndarray
        point at which the gradient is estimated
    a : numpy.ndarray
        generated argument points
    
    Return
    ----------
    res : numpy.ndarray
        estimated gradient at point x
    """

    m = len(a) # generated args num
    pol_degree = len(x)-1
    res = np.zeros((pol_degree+1, pol_degree+1), dtype=np.longdouble)
    for j in range(m):
        c = [a[j]**n for n in range(pol_degree+1)]
        res += np.outer(c, c)
    return res
