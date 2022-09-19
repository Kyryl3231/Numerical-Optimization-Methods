import numpy as np

def line_search_alpha(obj_fun, x, p, gradient, *obj_fun_args, **kwargs):
    """
    Find step-length using backtracking line search

    Params
    ----------
    obj_fun: function
        objective function
    x : numpy.longdouble or numpy.ndarray
        point for which step-length is estimated 
    p : numpy.longdouble or numpy.ndarray
        step direciton
    gradient: np.longdouble or numpy.ndarray
        gradient of obj_fun at the point x
    *obj_fun_args
        additional arguments for objective function
    
    Additional params
    ----------
    a_0 : numpy.longdouble
        initial step-length
    rho : numpy.longdouble
        contraction factor
    c : numpy.longdouble
        slope constant

    Return
    ----------
    alpha : numpy.longdouble
        step-length
    """

    a_0 = kwargs.get('a_0', np.longdouble(1.0))
    rho = kwargs.get('rho', np.longdouble(0.5))
    c = kwargs.get('c', np.longdouble(0.4))
    
    alpha = np.longdouble(a_0)
    while obj_fun(x+alpha*p, *obj_fun_args) > obj_fun(x, *obj_fun_args) + c*alpha*np.dot(gradient,p):
        alpha *= rho
    return alpha