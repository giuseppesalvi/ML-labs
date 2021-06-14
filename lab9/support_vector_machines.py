import numpy as np
import sklearn.datasets as da
import scipy.optimize as op
from modules.gaussian_models import split_db_2to1


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

        # z = np.array(2 * LTR - 1).reshape(1, N)
        z = np.array(2 * LTR - 1).reshape(N, 1)

        D = np.vstack((DTR, np.ones(N) * K))

        G = np.dot(D.T, D)
        H = z * z.T * G

        J_D = -1/2 * np.dot(np.dot(alpha.T, H), alpha) + \
            np.dot(alpha.T, np.ones(N))

        L_D = - J_D
        grad_L_D = (np.dot(H, alpha) - np.ones(N)).reshape(N, 1)

        return L_D, grad_L_D
    return svm_dual


def svm_primal_from_dual(alpha_s, DTR, LTR, K):
    """
    """
    N = LTR.shape[0]
    # z = np.array(2 * LTR - 1).reshape(1, N)
    # D = np.vstack((DTR, np.ones(N) * K))
    # w_s = np.dot(np.dot(alpha_s, z), D).sum
    
    z = np.array(2 * LTR - 1).reshape(N, 1)
    D = np.vstack((DTR, np.ones(N) * K))
    w_s = np.sum(alpha_s * z * D.T, axis = 0) # w_s with hat ^
    return w_s

def svm_primal_objective(w_s, DTR, LTR, K, C):
    """
    """
    N = LTR.shape[0]
    # z = np.array(2 * LTR - 1).reshape(N, 1)
    z = np.array(2 * LTR - 1).reshape(1, N)
    D = np.vstack((DTR, np.ones(N) * K))
    w_s = w_s.reshape((w_s.size, 1))
    f = 1 - z * np.dot(w_s.T, D)
    J = 1/2 * (w_s * w_s).sum() + C * np.sum(np.maximum(np.zeros(f.shape), f)) 
    return J


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
            w_s = svm_primal_from_dual(x.reshape(x.size, 1), DTR, LTR, K)
            # w_s = svm_primal_from_dual(x.reshape(x.shape[0], 1), DTR, LTR, K)

            # Compute scores S
            w_s_ = w_s[0:-1] # w star without hat ^
            b_s = w_s[-1]
            S = np.dot(w_s_.reshape(w_s_.size, 1).T, DTE) + b_s

            # Assign pattern comparing scores with threshold = 0
            predictions = 1 * (S > 0)
            
            # Compute accuracy and error rate
            correct_p= (predictions == LTE).sum()
            wrong_p= predictions.size - correct_p
            accuracy = correct_p / predictions.size
            error = wrong_p / predictions.size

            primal_obj = svm_primal_objective(w_s, DTR, LTR, K, C)
            dual_obj = -f
            duality_gap = primal_obj - dual_obj 
            print("K: %d, C : %.1f, Primal loss: %5e  Dual loss: %5e  Duality gap: %5e  Error rate: %.1f" % (K, C, primal_obj, dual_obj, duality_gap, error*100), "%")
