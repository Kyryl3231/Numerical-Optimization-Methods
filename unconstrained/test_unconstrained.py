import numpy as np
import matplotlib.pyplot as plt
import test_functions as tf
from unconstrained import steepest_descent, newton_method
import sys
sys.path.append("..")
from utilities.gradient import gradient_ls
from utilities.hessian import hessian_ls

criteria = ["epsilon * |f(x_k)-f(x_k-1)| >= |f(x_k+1) - f(x_k)|",
            "epsilon ||x_k-x_k-1|| >= ||x_k+1 - x_k||",
            "epsilon >= ||gradient f(x_k)||", 
            "number of iterations"]

optimization_methods = [steepest_descent, newton_method]
optimization_methods_names = ["The Steepest Descent", "Newton Method"]
for opt_fun, opt_fun_name in zip(optimization_methods, optimization_methods_names):            
    print(opt_fun_name)
    # TEST 1
    for fun,sol,name in zip(tf.test1_fun_list, tf.test1_solutions, tf.test1_fun_names):
        print("Function: {}".format(name))
        for x_0 in tf.test1_starting_points:
            x_0 = np.longdouble(x_0)
            minimizer, iter_count, stopping_criterion, stopping_values = opt_fun(fun, x_0, max_iter=10**5)
            dist_to_solution = min([np.linalg.norm(minimizer-x) for x in sol])
            print("Minimizer x = [{:.5f}, {:.5f}] was found in {} iterations. Distance to solution: {:.6f}".format(minimizer[0], minimizer[1], iter_count, dist_to_solution)) 
            print("Stopping criterion: {}".format(criteria[stopping_criterion]))
            for i in range(len(criteria)):
                print(criteria[i]," = ",stopping_values[i])
            print()

    # TEST 2
    plot_num = len(tf.test2_param_list)
    fig = plt.figure(figsize=(10,5))
    for plot_idx in range(plot_num):
        # a = np.random.random(size=m)*2*q - q # points on [-q, q]
        # a.sort()
        q, m, max_degree = tf.test2_param_list[plot_idx]
        a = np.linspace(-q,q,m)
        b = np.sin(a)
        x_0 = np.zeros(max_degree+1, dtype=np.longdouble)
        plt.subplot(1, plot_num, plot_idx+1)
        plt.plot(a,b, 'o-', label="original data")
        if opt_fun == newton_method:
            minimizer, it_num, stopping_criterion, stopping_values = opt_fun(tf.least_squares, x_0, a, b, gradient_fun=gradient_ls, hessian_fun=hessian_ls, max_iter=10**6)
        else:
            minimizer, it_num, stopping_criterion, stopping_values = opt_fun(tf.least_squares, x_0, a, b, gradient_fun=gradient_ls, max_iter=10**6)
        print("Result: {}".format(minimizer))
        print("Number of iterations: {}".format(it_num))
        print("Stopping criterion: {}".format(criteria[stopping_criterion]))
        for i in range(len(criteria)):
            print(criteria[i]," = ",stopping_values[i])
        print("Function to approximate: g(x) = sin(x)")
        print("Maximal degree of approximation: {}".format(max_degree))
        pol_str = " + ".join(["{:.5f}*x^{}".format(minimizer[i],i) for i in range(len(minimizer))])
        print("Final optimal polynomial:")
        print(pol_str, '\n')
        d = np.array([tf.phi(minimizer, el) for el in a]) # approximation
        taylor = np.array([tf.taylor_expansion_sin(el,max_degree) for el in a])
        plt.plot(a,d, 'o-', label="approximation")
        plt.plot(a,taylor, 'o-', label="taylor expansion")
        plt.xlabel("argument value")
        plt.ylabel("sin() value")
        plt.ylim(-1,1)
        plt.title("[-{}, {}] with {} points and max_degree = {}".format(q,q,m,max_degree))
        plt.legend()
    plt.show()
    fig.savefig('test_plots')