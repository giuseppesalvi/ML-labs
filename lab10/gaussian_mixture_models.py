from matplotlib.pyplot import axis
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
from data.GMM_load import load_gmm
from modules.gaussian_density_estimation import logpdf_GAU_ND
from modules.gaussian_density_estimation import GAU_logpdf 

def vcol(x):
    """ reshape the vector x into a column vector """

    return x.reshape(x.shape[0], 1)


def vrow(x):
    """ reshape the vector x into a row vector """

    return x.reshape(1, x.shape[0])

def mcol(x):
    """ reshape row vector into col vector"""
    return x.reshape(x.shape[1], 1)

def logpdf_GMM(X, gmm):
    """ Computes the log-density of a gmm for a set of samples contained in 
        matrix X
        X is a matrix of samples of shape (D, N),
        where D is the size of a sample 
        and D is the number of samples in X
        gmm is a list of component parameters representing the GMM:
        gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
        the result will be an array of shape (N,), whose components will
        contain the log-density for sample xi
    """

    M = len(gmm)
    N = X.shape[1]

    # Each row of S contains the (sub-)class conditional densities given
    # component Gi = g for all samples xi
    S = np.empty([M, N])

    for g in range(len(gmm)):
        for j, sample in enumerate(X.T):
            sample = vcol(sample)
            # gmm[g][1] = mu_g, gmm[g][2] = C_g
            S[g, j] = logpdf_GAU_ND(sample, vcol(gmm[g][1]), gmm[g][2])

    # Add to each row of S the logarithm of the prior of the corresponding
    # component log w_g
    for g in range(len(gmm)):
        # gmm[g][0] = w_g
        S[g, :] += np.log(gmm[g][0])

    # Compute the log-marginal log f_Xi(xi) for all samples xi
    logdens = sps.logsumexp(S, axis=0)

    return logdens


def EM_algorithm(X, initial_gmm, printDetails=False):
    """ Implementation of the GMM EM estimation procediure:
        the EM algorithm is useful to estimate the parameters of a GMM
        that maximize the likelihood for traning set X
        The initial estimate of the GMM is passed as parameter
        print is a boolean, to print iterations of the algorithm or no
    """

    gmm = initial_gmm
    M = len(gmm)
    F = X.shape[0] # number of features of each sample
    N = X.shape[1]
    stop = False

    # calculate the average loglikelihood using the initial gmm
    previous_avg_ll = sum(logpdf_GMM(X, initial_gmm)) / N

    if(printDetails):
        print("-"*40)
        print("\nEM algorithm starting\n")
        print("INITIAL       avg ll: ", previous_avg_ll)

    # continue the algorithm untill the stopping criterion is met
    counter = 1
    while(stop == False):

        # E-step

        # Each row of S contains the (sub-)class conditional densities given
        # component Gi = g for all samples xi
        S = np.empty([M, N])

        for g in range(M):  # for g in range(len(gmm)):
            for j, sample in enumerate(X.T):
                sample = vcol(sample)
                # gmm[g][1] = mu_g, gmm[g][2] = C_g
                S[g, j] = logpdf_GAU_ND(sample, vcol(gmm[g][1]), gmm[g][2])

        # Add to each row of S the logarithm of the prior of the corresponding
        # component log w_g
        for g in range(M):  # for g in range(len(gmm)):
            # gmm[g][0] = w_g
            S[g, :] += np.log(gmm[g][0])

        # S is now the matrix of joint densities f_Xi,Gi(xi,g)

        # Compute the log-marginal log f_Xi(xi) for all samples xi
        logdens = sps.logsumexp(S, axis=0)

        # Remove from each row of the joint densities matrix S the row vector
        # containing the N marginal densities logdens
        log_responsabilities = S - logdens

        # Compute the MxN matrix of class posterior probabilities, responsabilities
        responsabilities = np.exp(log_responsabilities)

        # M-step

        # Compute statistics
        Zg_list = []
        Fg_list = []
        Sg_list = []
        for g in range(M):
            Zg_list.append(vrow(responsabilities[g]).sum(axis=1))
            Fg_list.append((vrow(responsabilities[g]) * X).sum(axis=1))
            tmp = np.zeros((F, F))
            for i in range(N):
                tmp += responsabilities[g][i] * \
                    np.dot(vcol(X.T[i]), vrow(X.T[i]))
            Sg_list.append(tmp)

        # Obtain the new paramters
        for g in range(M):
            w_new = (Zg_list[g] / sum(Zg_list))[0]  # extract the float
            mu_new = vcol(Fg_list[g] / Zg_list[g])
            sigma_new = (Sg_list[g] / Zg_list[g]) - \
                np.dot(vcol(mu_new), vrow(mu_new))
            gmm[g] = (w_new, mu_new, sigma_new)

        # Check stopping criterion
        threshold = 1e-6
        this_avg_ll = sum(logpdf_GMM(X, gmm)) / N
        if printDetails:
            print("ITERATION ", counter, " avg ll: ", this_avg_ll)
        if (this_avg_ll - previous_avg_ll < threshold):
            stop = True
            if printDetails:
                print("STOPPING CRITERION MET")
                print("\nEM algorithm finished\n")
                print("-"*40)
        else:
            previous_avg_ll = this_avg_ll
            counter += 1

    return gmm

def plot_gmm(X, gmm):
    for x in X:
        plt.figure()
        plt.hist(np.sort(x), bins=30, density=True)
        plt.plot(np.sort(x), np.exp(logpdf_GMM(vrow(np.sort(x)), gmm)))

        plt.show()
    return 

if __name__ == "__main__":

    # Gaussian mixture models

    # Load data
    X = np.load("data/GMM_data_4D.npy")

    # Load the reference GMM
    gmm = load_gmm("data/GMM_4D_3G_init.json")

    # Solution log densities for all samples in X
    sol_logdens = np.load("data/GMM_4D_3G_init_ll.npy")

    # print("Solution log densities:")
    # print(sol_logdens)

    # Check if my algorithm is correct
    log_dens = logpdf_GMM(X, gmm)

    # print("My results:")
    # print(log_dens)

    # print("Difference: %f" % (sol_logdens - log_dens).sum())
    if (sol_logdens - log_dens).sum():
        print("Error: algorithm of GMM does not work")

    # GMM estimation: the EM algorithm

    # EM_gmm = EM_algorithm(X, gmm, printDetails=True)

    # Solution
    sol_EM_gmm = load_gmm("data/GMM_4D_3G_EM.json")

    # Check my results with solution
    # print("My results:")
    # print(EM_gmm)
    # print("Solution:")
    # print(sol_EM_gmm)

    # 1D case, plot the estimated density 
    X1D = np.load("data/GMM_data_1D.npy")
    gmm1D = load_gmm("data/GMM_1D_3G_init.json")

    EM_gmm1D = EM_algorithm(X1D, gmm1D, printDetails=True)


    plt.figure()
    plt.hist(mcol(np.sort(X1D)), bins=30, density=True)
    plt.plot(mcol(np.sort(X1D)), np.exp(logpdf_GMM(np.sort(X1D), EM_gmm1D)))
    plt.show()

