import numpy as np

# TEST 1
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2 

def difficult_fun(x):
    return 150*(x[0]*x[1])**2+(0.5*x[0]+2*x[1]-2)**2

test1_fun_list = [rosenbrock, difficult_fun]
test1_fun_names = ["Rosenbrock", "Difficult function"]
test1_solutions = np.array([[(1, 1)], [(0, 1), (4, 0)]], dtype=np.ndarray)
for i in range(len(test1_solutions)):
    test1_solutions[i] = np.array(test1_solutions[i])
test1_starting_points = np.longdouble([[-0.2,1.2], [3.8, 0.1], [0,0], [-1,0], [0,-1]])

# TEST 2
def phi(x: np.ndarray, t: np.longdouble):
    t_pol = [t**n for n in range(len(x))]
    return np.dot(x, t_pol)

def residual(x: np.ndarray, a: np.longdouble, b: np.longdouble):
    return phi(x, a) - b

def least_squares(x: np.ndarray, a: np.ndarray, b: np.ndarray):
    m = len(a)
    res = np.longdouble(0)
    for j in range(m):
        res += residual(x, a[j], b[j])**2
    return 0.5 * res

def taylor_expansion_sin(x, degree):
    degree -= not degree%2 # closest odd integer < degree
    n = int(np.floor(degree/2)) # maximal value of n
    res = np.longdouble(0)
    for i in range(n+1):
        res += ((-1)**i * (x)**(2*i+1)) / np.math.factorial(2*i+1)
    return res

# (q,m,max_degree)
# q - points are generated on [-q, q]
# m - number of generated arguments
test2_param_list = [(1, 50, 2), (2, 50, 3), (2.5, 100, 3), (3, 100, 3), (4, 100, 4)]