# Numerical-Optimization-Methods

This repository is dedicated to numerical optimization methods :magic_wand: :sparkles:.

## Unconstrained Optimization
All implemented methods are line search methods.
Implementations can be found in [`unconstrained/unconstrained.py`](unconstrained/unconstrained.py)

### The Steepest Descent
Optimization method based on following the direction of negative gradient.

### Newton Method
Optimization method that uses second order information.
Hessian matrix is estimated at every iteration.
In case it isn't positive definite, a multiple of identity matrix is added.

### Conjugate gradient
Optimization method based on conjugacy of step directions.
The `Polak–Ribiere` variation is implemented.
Update rule for a scalar that ensures conjugacy of step directions:
$\beta^{+}_{k+1} = max(0, \beta^{PR}_{k+1})$, where 

$\beta^{PR}_{k+1} = \frac{\nabla f^T_{k+1}(\nabla f_{k+1} - \nabla f_k)}{||\nabla f_k||^2}$
$f$ - objective function

### Quasi-Newton Method
Optimization method that uses an approximation of second order information.
The `BFGS` method is implemented.
The initial approximation of a Hessian matrix is an identity matrix.

### Testing
#### Problem 1: Optimize functions
1. $f(x) = 100(x_2-x_1^2)^2+(1-x_1)^2 \text{ || min at }x=(1,1)$
2. $f(x) = 150(x_1 x_2)^2+(0.5x_1+2x_2-2)^2 \text{ || min at }x=(0,1),(4,0)$

**Starting points:** (-0.2,1.2), (0,0), (-1,0)
#### Problem 2: Solve least square problems (approximate `sin(x)`)
1. **50** points are generated on **[-1,1]** interval
   approximation using `2nd` order polynomial
2. **100** points are generated on **[-3,3]** interval
   approximation using `3nd` order polynomial

### Demonstration

To see test results run `python -m unconstrained.test_unconstrained` in the base directory.
Text output will be written to the `unconstrained/test_unconstrained_results.txt`
Graphical representation of the solutions for least square problems can be found at `unconstrained/[optimization_method_name].png` (i.e. `unconstrained/The Steepest Descent.png`)

## Utilities
Here you can find some details about additional implemented functions.

### Gradient Estimation

**Central-difference formula was used**
$\frac{\partial{f}}{\partial{x_i}}(x) = \frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon} \\$ 

Float data type used: `numpy.longdouble`

On Windows with $\epsilon = 10^{-5}$
On Linux (Ubuntu 18.04.2 LTS) with $\epsilon = 10^{-6}$
*Reason*: on Windows the highest possible precision of float datatype is 15, on Linux - 18 <br>
Recommendation from the Book (p.197) $\epsilon = \textbf{u}^{\frac{1}{3}}$
$\textbf{u}$ - unit roundoff (bound on the relative error that is introduced whenever an arithmetic
operation is performed on two floating-point numbers)

**Another option: forward-difference**

$\frac{\partial{f}}{\partial{x_i}}(x) = \frac{f(x+\epsilon e_i)-f(x)}{\epsilon} \\$

Recommendation from the Book $\epsilon = \sqrt{\textbf{u}}$
On Windows with $\epsilon = 10^{-7}$
On Linux (Ubuntu 18.04.2 LTS) with $\epsilon = 10^{-9} \\$

### Hessian Matrix Estimation
Used formula

$\frac{\partial^2 f(x)}{\partial x_i \partial x_j}
= \frac{f(x+\epsilon e_i + \epsilon e_j) - f(x+\epsilon e_i) - f(x+\epsilon e_j) + f(x)}{\epsilon^2}$

### Step-Length line search
Used technique - ***Backtracking Line Search***

Line search algorithm for the Wolfe conditions can also be used `[is preffered]`.

## References
<ul>
  <li> :book: "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright

  Link: https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf </li>
</ul> 