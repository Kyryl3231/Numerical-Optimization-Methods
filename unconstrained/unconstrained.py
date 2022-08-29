import numpy as np
from typing import Callable
import sys
sys.path.append("..")

from utilities.gradient import gradient
from utilities.step_length import line_search_alpha


def steepest_descent(obj_fun: Callable, x_0: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    The steepest descent optimization

    Params
    ----------
    obj_fun : function
        objective function to minimize
    x_0 : numpy.mdarray
        initial point
    *args: 
        additional arguments for obj_fun

    Additional params
    ----------
    max_iter : int 
        maximum number of iterations
    tol : np.longdouble
        tolerance for the solution
    gradient_fun : function
        gradient implementation
        
    Return
    ----------
    x : numpy.longdouble or numpy.ndarray
        minimizer of the objective function
    iter : int
        iteration counter
    stopping_criterion : int
        stopping criterion index
    stopping_values : numpy.ndarray
        values of stopping criterions
    """

    max_iter = kwargs.get('max_iter',1000)
    tol = kwargs.get('tol',np.longdouble(1e-5))
    gradient_fun = kwargs.get('gradient_fun',gradient)

    if isinstance(x_0,np.longdouble):
        dim_num = 1
    elif isinstance(x_0, np.ndarray):
        if len(x_0.shape) != 1:
            raise ValueError("x_0 should have only 1 dimension, got {}".format(len(x_0.shape)))
        if isinstance(x_0[0],np.longdouble):
            dim_num = len(x_0)
        else:
            raise TypeError("Elements of x_0 should be of type np.longdouble, got {}".format(type(x_0[0])))
    else:
        raise TypeError("x_0 should be of type numpy.longdouble or numpy.ndarray, got {}".format(type(x_0)))

    x = np.longdouble(x_0)
    p = np.ones(dim_num) # step direction
    iter = 0
    stopping_values = np.zeros(4)
    a_0 = np.longdouble(1.0)
    # norm_gradient_zero = np.linalg.norm(gradient_fun(obj_fun, x_0, *args))
    to_stop=False
    while not to_stop:
        # estimate gradient at x
        # find new p
        grad_new = gradient_fun(obj_fun, x, *args)
        p = -grad_new
        # find corresponding step size
        if iter>0:
          a_0 = alpha*(np.dot(grad,grad)/np.dot(grad_new,grad_new))
        alpha = line_search_alpha(obj_fun, x, p, grad_new,*args,rho=0.7, a_0=a_0)
        # update iterations counter
        iter += 1
        grad = grad_new
        stopping_values[2] = np.linalg.norm(grad)
        stopping_values[3] = iter
        # update x
        x_new = x + alpha*p
        fun_value_step_new = np.abs(obj_fun(x_new, *args)-obj_fun(x, *args))
        arg_step_new = np.linalg.norm(x_new-x)
        x = x_new 
        # check stopping criteria
        if fun_value_step_new <= tol*stopping_values[0]:
            stopping_criterion = 0
            to_stop = True
        if arg_step_new <= tol*stopping_values[1]:
            stopping_criterion = 1
            to_stop = True
        if stopping_values[2] <= tol:
            stopping_criterion = 2
            to_stop = True
        if iter>=max_iter:
            stopping_criterion = 3
            to_stop = True    
        stopping_values[0] = fun_value_step_new
        stopping_values[1] = arg_step_new

    return x, iter, stopping_criterion, stopping_values

