import numpy as np
from typing import Callable
import scipy
import sys
sys.path.append("..")

from utilities.gradient import gradient
from utilities.hessian import hessian
from utilities.step_length import line_search_alpha

def check_stopping_criteria(stopping_values, fun_step_new, arg_step_new, tol, max_iter, iter):
    """
    Check stopping criteria for optimization

    Params
    ----------
    stopping_values : numpy.ndarray
        values of stopping criterions 
    fun_step_new : float
        difference in objective function value btw x_new and x
    arg_step_new : float
        difference btw x_new and x
    tol : np.longdouble
        tolerance for the solution
    max_iter : int 
        maximum number of iterations
    iter : int
        current iteration number

    Return
    ----------
    to_stop : bool
        stop flag for optimization
    stopping_criterion : int
        stopping criterion index
    """

    if fun_step_new <= tol*stopping_values[0]:
        to_stop = True
        stopping_criterion = 0
        return to_stop, stopping_criterion
    if arg_step_new <= tol*stopping_values[1]:
        to_stop = True
        stopping_criterion = 1
        return to_stop, stopping_criterion
    if stopping_values[2] <= tol:
        to_stop = True
        stopping_criterion = 2
        return to_stop, stopping_criterion
    if iter>=max_iter:
        to_stop = True
        stopping_criterion = 3
        return to_stop, stopping_criterion
    return False, None

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
        to_stop, stopping_criterion = check_stopping_criteria(stopping_values, fun_value_step_new,
                                                              arg_step_new, tol, max_iter, iter)
        stopping_values[0] = fun_value_step_new
        stopping_values[1] = arg_step_new

    return x, iter, stopping_criterion, stopping_values


def cholesky_modified(A: np.ndarray, beta=np.longdouble(1e-3), lower=False):
    """
    Modified Cholesky factorization with added multiple of identity

    Params
    ----------
    A : numpy.ndarray
        matrix to decompose
    beta : np.longdouble
        modification parameter
    lower : bool
        Whether to compute the upper- or lower-triangular Cholesky factorization
    
    Return
    ----------
    L : numpy.ndarray
        Cholesky factor 
    """

    el_num = len(A)
    min_diag_el = min([A[i,i] for i in range(el_num)])
    if min_diag_el > 0:
        tau = np.longdouble(0)
    else:
        tau = -min_diag_el + beta
    
    while True:
        try: 
            L = scipy.linalg.cholesky(A+tau*np.identity(el_num), lower=lower)
            break
        except scipy.linalg.LinAlgError:
            tau = max(2*tau, beta)
    return L

def newton_method(obj_fun: Callable, x_0: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    The Newton Method optimization

    Params
    ----------
    obj_fun : function
        objective function to minimize
    x_0 : numpy.ndarray
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
        implementation for gradient estimation
    hessian_fun: function
        implementation for hessian estimation
        
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
    hessian_fun = kwargs.get('hessian_fun',hessian)

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
    grad = gradient_fun(obj_fun, x, *args)
    iter = 0
    stopping_values = np.zeros(4)
    to_stop=False
    while not to_stop:
        grad = gradient_fun(obj_fun, x, *args)
        hess = hessian_fun(obj_fun, x, *args)
        if dim_num==1:
            p = -hess**(-1)*grad
        else:
            # get p by solving hess @ p = -grad 
            # try:
            # cholesky decomposition of the Hessian
            L = cholesky_modified(hess, lower=True)
            # solve L @ L.T @ p = -grad with L.T @ p = z
            z = scipy.linalg.solve(L,-grad, lower=True)
            p = scipy.linalg.solve(L.T, z, lower=False)
            # except scipy.linalg.LinAlgError:
                # p = -grad
                # print("grad was used")
        alpha = line_search_alpha(obj_fun, x, p, grad, *args) 
        iter += 1
        stopping_values[2] = np.linalg.norm(grad)
        stopping_values[3] = iter
        # update x
        x_new = x + alpha*p
        fun_value_step_new = np.abs(obj_fun(x_new, *args)-obj_fun(x, *args))
        arg_step_new = np.linalg.norm(x_new-x)
        x = x_new 
        # check stopping criteria
        to_stop, stopping_criterion = check_stopping_criteria(stopping_values, fun_value_step_new,
                                                              arg_step_new, tol, max_iter, iter)
        stopping_values[0] = fun_value_step_new
        stopping_values[1] = arg_step_new
    return x, iter, stopping_criterion, stopping_values


def conjugate_gradient(obj_fun: Callable, x_0: np.ndarray, * args, **kwargs) -> np.ndarray:
    """
    PR+ Conjugate Gradient optimization

    Params
    ----------
    obj_fun : function
        objective function to optimize
    x_0 : numpy.ndarray
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
        implementation for gradient estimation

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
    grad = gradient_fun(obj_fun, x, *args)
    iter = 0
    p = -grad
    stopping_values = np.zeros(4)
    to_stop=False
    while not to_stop:
        alpha = line_search_alpha(obj_fun, x, p, grad, *args) # step size
        x_new = x+alpha*p
        stopping_values[2] = np.linalg.norm(grad)
        stopping_values[3] = iter
        
        grad_new = gradient_fun(obj_fun, x_new, *args)
        beta = np.dot(grad_new, grad_new-grad)/np.dot(grad, grad)
        beta = max(0, beta)
        p = -grad_new + beta*p
        iter += 1
        fun_value_step_new = np.abs(obj_fun(x_new, *args)-obj_fun(x, *args))
        arg_step_new = np.linalg.norm(x_new-x)
        stopping_values[2] = np.linalg.norm(grad)
        stopping_values[3] = iter
        x = x_new
        grad = grad_new
        # check stopping criteria
        to_stop, stopping_criterion = check_stopping_criteria(stopping_values, fun_value_step_new,
                                                              arg_step_new, tol, max_iter, iter)
        stopping_values[0] = fun_value_step_new
        stopping_values[1] = arg_step_new

    return x, iter, stopping_criterion, stopping_values


def quasi_newton_method(obj_fun: Callable, x_0: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    The BFGS Quasi-Newton Method optimization

    Params
    ----------
    obj_fun : function
        objective function to optimize
    x_0 : numpy.ndarray
        a vector representing initial point
    *args: 
        additional arguments for obj_fun

    Additional params
    ----------
    max_iter : int 
        maximum number of iterations
    tol : np.longdouble
        tolerance for the solution
    gradient_fun : function
        implementation for gradient estimation

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
    grad = gradient_fun(obj_fun, x, *args)
    if dim_num == 1:
        hess = 1
    else:
        hess = np.identity(n=dim_num, dtype=np.longdouble)
    iter = 0
    stopping_values = np.zeros(4)
    to_stop=False
    while not to_stop:
        if dim_num == 1:
            p = -hess * grad
        else:
            p = -hess @ grad 
        
        alpha = line_search_alpha(obj_fun, x, p, grad, *args, c=1e-4, rho=0.9)
        x_new = x + alpha*p
        grad_new = gradient_fun(obj_fun, x_new, *args)
        s = x_new - x
        y = grad_new - grad
        rho = 1/(np.dot(y,s))
        if iter == 0: # scale H before the first update
            hess *= np.dot(y,s)/np.dot(y,y)
        if dim_num == 1:
            hess = (1-rho*s*y)*hess*(1-rho*y*s)+rho*s*s 
        else:
            I = np.identity(dim_num)
            hess = (I-rho*np.outer(s,y))@hess@(I-rho*np.outer(y,s))+rho*np.outer(s,s)
        iter += 1
        fun_value_step_new = np.abs(obj_fun(x_new, *args)-obj_fun(x, *args))
        arg_step_new = np.linalg.norm(x_new-x)
        stopping_values[2] = np.linalg.norm(grad_new)
        stopping_values[3] = iter
        x = x_new
        grad = grad_new
        # check stopping criteria
        to_stop, stopping_criterion = check_stopping_criteria(stopping_values, fun_value_step_new,
                                                              arg_step_new, tol, max_iter, iter)
        stopping_values[0] = fun_value_step_new
        stopping_values[1] = arg_step_new
    return x, iter, stopping_criterion, stopping_values
