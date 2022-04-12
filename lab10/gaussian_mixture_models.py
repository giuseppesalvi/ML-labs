import numpy as np
import scipy.special as sps
from modules.gaussian_density_estimation import logpdf_GAU_ND
from modules.pca_lda import covariance_matrix2
import sklearn.datasets as da
import matplotlib.pyplot as plt
import json


def vcol(x):
    """ reshape the vector x into a column vector """

    return x.reshape(x.shape[0], 1)


def vrow(x):
    """ reshape the vector x into a row vector """

    return x.reshape(1, x.shape[0])


def mcol(x):
    """ reshape row vector into col vector: (1, N) -> (N, 1)"""
    # return x.reshape(x.shape[1], 1)
    return x.reshape((x.size, 1))


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


def EM_algorithm(X, initial_gmm, psi=0.01, printDetails=False, version="full"):
    """ Implementation of the GMM EM estimation procedure:
        the EM algorithm is useful to estimate the parameters of a GMM
        that maximize the likelihood for traning set X
        The initial estimate of the GMM is passed as parameter
        psi is used to constrain the eigenvalues of covariance matrices 
        in order to avoid degenerate solutions
        printDetails is a boolean, to print iterations of the algorithm or no
        type can be "full", "diagonal", "tied" to specify which version to use
        the default is Full covariance
    """

    gmm = initial_gmm
    M = len(gmm)
    F = X.shape[0]  # number of features of each sample
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

        sum_covariances = np.zeros((F, F))  # used for tied covariance version
        # Obtain the new paramters
        for g in range(M):
            w_new = (Zg_list[g] / sum(Zg_list))[0]  # extract the float
            mu_new = vcol(Fg_list[g] / Zg_list[g])
            sigma_new = (Sg_list[g] / Zg_list[g]) - \
                np.dot(vcol(mu_new), vrow(mu_new))

            # diagonal version
            if(version == "diagonal"):
                sigma_new = sigma_new * np.eye(sigma_new.shape[0])

            # tied version
            if(version == "tied"):
                sum_covariances += Zg_list[g] * sigma_new

            # Constraining the eigenvalues of the covariance matrices to be
            # larger or equal to psi
            U, s, _ = np.linalg.svd(sigma_new)
            s[s < psi] = psi
            sigma_new = np.dot(U, vcol(s) * U.T)

            gmm[g] = (w_new, mu_new, sigma_new)

        for g in gmm:
            if(version == "tied"):
                g = (g[0], g[1], sum_covariances / N)

        # Check stopping criterion
        threshold = 1e-6
        this_avg_ll = sum(logpdf_GMM(X, gmm)) / N
        if printDetails:
            print("ITERATION ", counter, " avg ll: ", this_avg_ll)
        if (this_avg_ll - previous_avg_ll < threshold):
            stop = True
            if printDetails:
                print("\nSTOPPING CRITERION MET")
                print("\nEM algorithm finished\n")
                print("-"*40)
        else:
            previous_avg_ll = this_avg_ll
            counter += 1

    return gmm


# def plot_gmm(X, gmm):
#     for x in X:
#         plt.figure()
#         plt.hist(np.sort(x), bins=30, density=True)
#         plt.plot(np.sort(x), np.exp(logpdf_GMM(vrow(np.sort(x)), gmm)))
#         plt.show()
#     return

def LBG_algorithm(X, gmm=None, goal_components=None, alpha=0.1, psi=0.01, printDetails=False, version="full"):
    """ Implementation of the LBG algorithm:
        starting from a gmm passed as parameter (or GMM_1  if nothing is passed)
        incrementally constructs a GMM with 2G components from a GMM with G 
        components, we stop when we reach goal components
        type can be "full", "diagonal", "tied" to specify which version to use
        the default is Full covariance
    """

    if (gmm == None):
        # GMM_1 = [(w, mu, C)] = [(1.0, mu, C)] gaussian density
        gmm = [(1.0, vcol(X.mean(1)), covariance_matrix2(X))]

    components = len(gmm)

    # Constraining the eigenvalues of the covariance matrices
    # g[2] is the covariance matrix
    for g in gmm:
        U, s, _ = np.linalg.svd(g[2])
        s[s < psi] = psi
        g = (g[0], g[1], np.dot(U, vcol(s)*U.T))

    if (goal_components == None):
        goal_components = components * 2
    if (printDetails):
        print("-"*40)
        print("\nLBG algorithm starting\n")
    counter = 1
    while components < goal_components:
        if (printDetails):
            print("\nITERATION ", counter)
            print("to obtain n_components = ", components*2, "\n")
        new_gmm = []
        for g in gmm:

            # Constraining the eigenvalues of the covariance matrices
            # g[2] is the covariance matrix
            U, s, _ = np.linalg.svd(g[2])
            s[s < psi] = psi
            g = (g[0], g[1], np.dot(U, vcol(s)*U.T))

            U, s, Vh = np.linalg.svd(g[2])
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            new_gmm.append((g[0] / 2, g[1] + d, g[2]))
            new_gmm.append((g[0] / 2, g[1] - d, g[2]))

        # The 2G components gmm can be used as initial gmm for the EM algorithm
        gmm = EM_algorithm(X, new_gmm, psi, printDetails, version)
        counter += 1
        components *= 2

    if(printDetails):
        print("\nLBG algorithm finished\n")
        print("-"*40)

    return gmm


def load_iris():
    """ Load iris dataset from the sklearn library
        Returns the dataset matrix D and matrix of labels L
    """
    # We need to transpose the data matrix, since we work with column
    # representations of feature vectors
    D, L = da.load_iris()['data'].T, da.load_iris()['target']

    return D, L


def split_db_2to1(D, L, seed=0):
    """ Split the dataset in two parts, one is 2/3, the other is 1/3
        first part will be used for model training, second part for evaluation
        D is the dataset, L the corresponding labels
        returns:
        DTR = Dataset for training set
        LTR = Labels for training set
        DTE = Dataset for test set
        LTE = Labels for testset
    """
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def GMM_classifier(n_classes, DTR, LTR, DTE, LTE, M, psi, version):

    ld = []
    # for each class train GMM
    for i in range(n_classes):

        # keep only samples of ith class
        DTR_i = DTR[:, LTR == i]
        # apply LBG algorithm
        gmm = LBG_algorithm(DTR_i, goal_components=M, psi=psi,
                            version=version, printDetails=False)
        ld.append(logpdf_GMM(DTE, gmm))
    p = np.vstack((ld[0], ld[1], ld[2]))
    # compute predicted labels
    predicted = np.argmax(p, axis=0)
    correct = np.array(predicted == LTE).sum()
    accuracy = correct / LTE.size
    error_rate = 1.0 - accuracy
    return error_rate


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


if __name__ == "__main__":

    # Gaussian mixture models

    # Load data
    X = np.load("data/GMM_data_4D.npy")
    X1D = np.load("data/GMM_data_1D.npy")

    # Load the reference GMM
    gmm = load_gmm("data/GMM_4D_3G_init.json")
    gmm1D = load_gmm("data/GMM_1D_3G_init.json")

    # Solution log densities for all samples in X
    sol_logdens = np.load("data/GMM_4D_3G_init_ll.npy")

    # print("Solution log densities:")
    # print(sol_logdens)

    # Check if my algorithm is correct
    log_dens = logpdf_GMM(X, gmm)

    # print("My results:")
    # print(log_dens)

    # print("Difference: %f" % (sol_logdens - log_dens).sum())
    if abs((sol_logdens - log_dens).sum()) > 0.00000001:
        print("Error: algorithm of GMM does not work")

    # GMM estimation: the EM algorithm

    # 4D case, check result with solution
    print("4D dataset")
    EM_gmm = EM_algorithm(X, gmm, printDetails=True)

    # Solution
    sol_EM_gmm = load_gmm("data/GMM_4D_3G_EM.json")

    # Check my results with solution
    print("My results:")
    print(EM_gmm)
    print("Solution:")
    print(sol_EM_gmm)

    # 1D case, plot the estimated density

    print("1D dataset")
    EM_gmm1D = EM_algorithm(X1D, gmm1D, printDetails=True)

    plt.figure()
    plt.hist(mcol(np.sort(X1D)), bins=30, density=True)
    plt.plot(mcol(np.sort(X1D)), np.exp(logpdf_GMM(np.sort(X1D), EM_gmm1D)))
    plt.show()

    # LBG algorithm

    # 4D case, check results with solution
    print("4D dataset")
    print("Full Covariance")
    LBG_gmm = LBG_algorithm(X, goal_components=4,
                            printDetails=True, version="full")
    print("Diagonal Covariance")
    LBG_gmm = LBG_algorithm(X, goal_components=4,
                            printDetails=True, version="diagonal")
    print("Tied Covariance")
    LBG_gmm = LBG_algorithm(X, goal_components=4,
                            printDetails=True, version="tied")

    # # Solution
    sol_LBG_gmm = load_gmm("data/GMM_4D_4G_EM_LBG.json")
    # Check my results with solution: NB: results are correct but not sorted
    print("My results:")
    print(LBG_gmm)
    print("Solution:")
    print(sol_LBG_gmm)

    # 1D case, plot the estimated density
    print("1D dataset")
    LBG_gmm1D = LBG_algorithm(X1D, goal_components=4, printDetails=True)

    # Solution
    sol_LBG_gmm1D = load_gmm("data/GMM_1D_4G_EM_LBG.json")
    # Check my results with solution: NB: results are correct but not sorted
    print("My results:")
    print(LBG_gmm1D)
    print("Solution:")
    print(sol_LBG_gmm1D)

    plt.figure()
    plt.hist(mcol(np.sort(X1D)), bins=30, density=True)
    plt.plot(mcol(np.sort(X1D)), np.exp(logpdf_GMM(np.sort(X1D), LBG_gmm1D)))
    plt.show()

    # GMM for Classification
    print("GMM for classification")

    # Load iris dataset
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # Full Covariance
    M_list = [1, 2, 4, 8, 16]
    psi = 0.01

    versions = ["full", "diagonal", "tied"]
    for version in versions:
        error_rates = []
        for M in M_list:
            error_rates.append(GMM_classifier(
                3, DTR, LTR, DTE, LTE, M, psi, version))
        print("version " + version)
        for i, M in enumerate(M_list):
            print("M = %d, error_rate = %f" % (M, error_rates[i] * 100), "%")
        print("")
