import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from unconstrained.unconstrained import steepest_descent, newton_method, conjugate_gradient, quasi_newton_method
import unconstrained.test_functions as tf
from utilities.gradient import gradient_ls
from utilities.hessian import hessian_ls

criteria = ["epsilon * |f(x_k)-f(x_k-1)| >= |f(x_k+1) - f(x_k)|",
            "epsilon ||x_k-x_k-1|| >= ||x_k+1 - x_k||",
            "epsilon >= ||gradient f(x_k)||", 
            "number of iterations"]

optimization_methods = [steepest_descent, newton_method, conjugate_gradient, quasi_newton_method]
optimization_methods_names = ["The Steepest Descent", "Newton Method", "PR+ Conjugate Gradient", "BFGS Quasi-Newton Method"]

test_file = open('unconstrained/test_results.txt', 'w')

for opt_fun, opt_fun_name in zip(optimization_methods, optimization_methods_names):            
    test_file.writelines(['#'*10, '\n', opt_fun_name, '\n', '#'*10, '\n'*2])
    # TEST 1
    test_file.write('TEST #1\n\n')
    for fun,sol,name in zip(tf.test1_fun_list, tf.test1_solutions, tf.test1_fun_names):
        test_file.write("Function: {}\n".format(name))
        for x_0 in tf.test1_starting_points:
            x_0 = np.longdouble(x_0)
            minimizer, iter_count, stopping_criterion, stopping_values = opt_fun(fun, x_0, max_iter=10**5)
            dist_to_solution = min([np.linalg.norm(minimizer-x) for x in sol])
            test_file.write("Minimizer x = [{:.5f}, {:.5f}] was found in {} iterations. Distance to solution: {:.6f}\n".format(minimizer[0], minimizer[1], iter_count, dist_to_solution)) 
            test_file.write("Stopping criterion: {}\n".format(criteria[stopping_criterion]))
            for i in range(len(criteria)):
                test_file.write(criteria[i]+" = "+str(stopping_values[i])+'\n')
            test_file.write('\n')

    # TEST 2
    test_file.write('TEST #2\n\n')
    plot_num = len(tf.test2_param_list)
    fig = plt.figure(figsize=(10,5))
    fig.suptitle(opt_fun_name)
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
        test_file.write("Result: {}\n".format(minimizer))
        test_file.write("Number of iterations: {}\n".format(it_num))
        test_file.write("Stopping criterion: {}\n".format(criteria[stopping_criterion]))
        for i in range(len(criteria)):
            test_file.write(criteria[i]+" = "+str(stopping_values[i])+'\n')
        test_file.write("Function to approximate: g(x) = sin(x)\n")
        test_file.write("Maximal degree of approximation: {}\n".format(max_degree))
        pol_str = " + ".join(["{:.5f}*x^{}".format(minimizer[i],i) for i in range(len(minimizer))])
        test_file.write("Final optimal polynomial:\n")
        test_file.write(pol_str+'\n'*2)
        d = np.array([tf.phi(minimizer, el) for el in a]) # approximation
        taylor = np.array([tf.taylor_expansion_sin(el,max_degree) for el in a])
        plt.plot(a,d, 'o-', label="approximation")
        plt.plot(a,taylor, 'o-', label="taylor expansion")
        plt.xlabel("argument value")
        plt.ylabel("sin() value")
        plt.ylim(-1,1)
        plt.title("[-{}, {}] with {} points and max_degree = {}".format(q,q,m,max_degree))
        plt.legend()
    # plt.show()
    fig.savefig('unconstrained/'+opt_fun_name)

test_file.close()