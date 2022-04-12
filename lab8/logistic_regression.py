import numpy as np
import scipy.optimize as op
import sklearn.datasets as da
from modules.gaussian_models import split_db_2to1


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


def load_iris_binary():
    D, L = da.load_iris()['data'].T, da.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def logreg_obj_wrap(DTR, LTR, l):
    """ It's a wrapper for function logreg_obj that needs to access also to
        DTR, LTR, and lambda (l)
    """
    def logreg_obj(v):
        """ Computes the Logistic Regression objective J(w, b) using formula (2)
            v is a numpy array with shape (D+1,), where D is the dimensionality of 
            the feature space v = [w,b]
        """
        w, b = v[0:-1], v[-1]
        z = 2 * LTR - 1
        J = l / 2 * (w * w).sum() + np.log1p(np.exp(-z * (w.T.dot(DTR) + b))).mean()

        # using formula 3
        # J = l / 2 * (w * w).sum() + (LTR * np.log1p(np.exp(-w.T.dot(DTR) - b)) + (1 - LTR) * np.log1p(np.exp(w.T.dot(DTR) + b))).mean()

        return J

    return logreg_obj


if __name__ == "__main__":

    # 1. Numerical optimization

    x, f, d = op.fmin_l_bfgs_b(f1, np.array([0, 0]), approx_grad=True)
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

    x, f, d = op.fmin_l_bfgs_b(f2, np.array([0, 0]))
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

    # Load iris dataset, binary version
    D, L = load_iris_binary()

    # Split dataset in Training and Evaluation
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # starting point
    x0 = np.zeros(DTR.shape[0] + 1)

    # lambda is a hyper-parameter, we try different values and see how it
    # affects the performance
    lambda_list = [0., 1e-6, 1e-3, 1.]

    for l in lambda_list:
        # Use the wrapper to pass the parameters to the function
        logreg_obj = logreg_obj_wrap(DTR, LTR, l)

        # Obtain the minimizer of J
        x, f, d = op.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)

        # Now that the model is trained we can compute posterior
        # log-likelihoods ratios by simply computing for each test sample xt
        # the score s(xt) = wT xt + b
        # compute the array of scores S
        S = np.dot(x[0:-1].T, DTE) + x[-1]

        # Compute the class assignments by thresholding the scores with 0
        # LP is the array po predicted labels
        LP = (S > 0).astype(int)

        # Compute accuracy and error rate
        acc = (LP == LTE).mean()
        err = (1 - acc) * 100

        print("\n------------------------------------------------")
        print("Binary logistic regression")
        print("lambda : ", l)
        print("------------------------------------------------\n")
        print("estimated position for the minimum : ", "w* = (%.3f, %.3f, %.3f, %.3f)" % (x[0],x[1],x[2],x[3]), "b* = %.3f" % x[-1])
        print("objective value at the minimum J(w*, b*): %.5e" % f)
        print("number of iterations: ", d['nit'])
        print("number of times function f was called", d['funcalls'])
        print("error rate : %.1f" % err, "%")
        print("\n------------------------------------------------\n")