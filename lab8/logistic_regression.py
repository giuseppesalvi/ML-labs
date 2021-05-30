import numpy as np
import scipy.optimize as op 


def f1(x):
    """ f1 implements and returns the function:
        f(y, z) = (y + 3)² + sin(y) + (z + 1)²
        x is a 1-D numpy array of shape (2,)
        the first component of x corresponds to y, the second to z
    """
    y = x[0]
    z = x[1]
    return (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2


def f2(x):
    """ f1 implements and returns the function f(y, z) 
        and its gradient as a numpy array with shape (2,)
        f(y, z) = (y + 3)² + sin(y) + (z + 1)²
        x is a 1-D numpy array of shape (2,)
        the first component of x corresponds to y, the second to z
    """
    y = x[0]
    z = x[1]
    f = (y + 3) ** 2 + np.sin(y) + (z + 1) ** 2
    grad_f = np.array([2 * (y + 3) + np.cos(y), 2 * (z + 1)]) 
    return f, grad_f 

if __name__ == "__main__":

    # 1. Numerical optimization

    x, f, d = op.fmin_l_bfgs_b(f1, np.array([0,0]), approx_grad=True)
    print("\n------------------------------------------------")
    print("Numerical optimization") 
    print("Let the implmentation compute an approximated gradient")
    print("------------------------------------------------\n")
    print("function: f(y, z) = (y + 3)² + sin(y) + (z + 1)²")
    print("starting point: [0,0]")
    print("estimated position for the minimum: ", x)
    print("objective value at the minimum: ", f)
    print("number of iterations: ", d['nit']) 
    print("number of times function f was called", d['funcalls'])
    print("gradient ad the minimum, should be 0ish: ", d['grad'])
    print("\n------------------------------------------------\n")

    x, f, d = op.fmin_l_bfgs_b(f2, np.array([0,0]))
    print("\n------------------------------------------------")
    print("Numerical optimization") 
    print("Passing the function and its gradient manually")
    print("------------------------------------------------\n")
    print("function: f(y, z) = (y + 3)² + sin(y) + (z + 1)²")
    print("starting point: [0,0]")
    print("estimated position for the minimum: ", x)
    print("objective value at the minimum: ", f)
    print("number of iterations: ", d['nit']) 
    print("number of times function f was called", d['funcalls'])
    print("gradient ad the minimum, should be 0ish: ", d['grad'])
    print("\n------------------------------------------------\n")

    # 2. Binary logistic regression