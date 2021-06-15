import numpy as np
import scipy.special as sps
from data.GMM_load import load_gmm
from modules.gaussian_density_estimation import logpdf_GAU_ND

def vcol(x):
    """ reshape the vector x into a column vector """

    return x.reshape(x.shape[0], 1)


def logpdf_GMM(X, gmm):
    """ Computes the log-density of a gmm for a set of samples contained in 
        matrix X
        X is a matrix of samples of shape (D, N),
        where D is the size of a sample 
        and D is the number of samples in X
        gmm is a list of component parameters representing the GMM:
        gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
        the result will be an array of shape (N,), whose components i will
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


if __name__ == "__main__":

    # Gaussian mixture models
    
    # Load data 
    X = np.load("data/GMM_data_4D.npy")

    # Load the reference GMM 
    gmm = load_gmm("data/GMM_4D_3G_init.json")

    # Solution log densities for all samples in X
    sol_logdens = np.load("data/GMM_4D_3G_init_ll.npy")

    # print("SOLUTION log densities:")
    # print(sol_logdens)

    # Check if my algorithm is correct
    log_dens = logpdf_GMM(X, gmm) 

    # print("My results:")
    # print(log_dens)

    # print("Difference: %f" % (sol_logdens - log_dens).sum())
    if (sol_logdens - log_dens).sum():
        print("Error: algorithm of GMM does not work")


    # GMM estimation: the EM algorithm
    