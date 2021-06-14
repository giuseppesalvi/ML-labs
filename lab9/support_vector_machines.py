import numpy as np
import sklearn.datasets as da
import scipy.optimize as op
from modules.gaussian_models import split_db_2to1


def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))


def load_iris_binary():
    D, L = da.load_iris()['data'].T, da.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def svm_dual_wrapper(DTR, LTR, K):
    """
    """
    def svm_dual(alpha):
        """
        """
        N = DTR.shape[1]

        z = mcol(np.array(2 * LTR - 1))

        D_hat = np.vstack((DTR, np.ones(N) * K))
        G_hat = np.dot(D_hat.T, D_hat)
        H_hat = z * z.T * G_hat

        # J_D_hat = -1/2 * np.dot(np.dot(alpha.T, H_hat), alpha) + \
        #     np.dot(alpha.T, np.ones(N))

        # need to use multi_dot becuase it gives numerical problems otherwise
        J_D_hat = -1/2 * np.linalg.multi_dot([alpha.T, H_hat, alpha]) + \
            np.dot(alpha.T, np.ones(N))

        L_D_hat = - J_D_hat
        grad_L_D_hat = mcol(np.dot(H_hat, alpha) - np.ones(N))

        return L_D_hat, grad_L_D_hat
    return svm_dual


def svm_primal_from_dual(alpha, DTR, LTR, K):
    """
    """
    N = LTR.shape[0]
    z = mcol(np.array(2 * LTR - 1))
    D = np.vstack((DTR, np.ones(N) * K))
    w_s_hat = np.sum(alpha * z * D.T, axis=0)  
    return w_s_hat


def svm_primal_objective(w_s_hat, DTR, LTR, K, C):
    """
    """
    N = LTR.shape[0]
    z = mrow(np.array(2 * LTR - 1))
    D_hat = np.vstack((DTR, np.ones(N) * K))
    f = 1 - z * np.dot(w_s_hat.T, D_hat)
    J_hat = 1/2 * (w_s_hat * w_s_hat).sum() + C * \
        np.sum(np.maximum(np.zeros(f.shape), f))
    return J_hat


if __name__ == "__main__":

    # Load iris dataset, binary version
    D, L = load_iris_binary()

    # Split dataset in Training and Evaluation
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    N = DTR.shape[1]

    x0 = np.zeros(N)

    K_list = [1, 10]
    C_list = [0.1, 1.0, 10.0]
    for K in K_list:
        svm_dual = svm_dual_wrapper(DTR, LTR, K)
        for C in C_list:

            bounds = []
            for i in range(N):
                bounds.append((0, C))

            x, f, d = op.fmin_l_bfgs_b(svm_dual, x0, factr=1.0, bounds=bounds)

            # Recover primal solution from dual solution
            w_s_hat = svm_primal_from_dual(mcol(x), DTR, LTR, K)

            # Compute scores S

            # option 1: doesn't work
            # w_s = w_s_hat[0:-1]
            # b_s = w_s_hat[-1]
            # S = np.dot(mcol(w_s).T, DTE) + b_s

            # DTE_ is the extended data matrix for the evaluation set
            DTE_ = np.vstack((DTE, np.ones(DTE.shape[1]) * K))
            S = np.dot(mcol(w_s_hat).T, DTE_)

            # Assign pattern comparing scores with threshold = 0
            predictions = 1 * (S > 0)

            # Compute accuracy and error rate
            correct_p = (predictions == LTE).sum()
            wrong_p = predictions.size - correct_p
            accuracy = correct_p / predictions.size
            error = wrong_p / predictions.size

            primal_obj = svm_primal_objective(mcol(w_s_hat), DTR, LTR, K, C)
            dual_obj = -f
            duality_gap = primal_obj - dual_obj
            print("K: %d, C : %.1f, Primal loss: %5e  Dual loss: %5e  Duality gap: %5e  Error rate: %.1f" % (
                K, C, primal_obj, dual_obj, duality_gap, error*100), "%")
