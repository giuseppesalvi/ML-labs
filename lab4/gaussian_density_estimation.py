import numpy as np
import matplotlib.pyplot as plt


def GAU_pdf(x, mu, var):
    """ computes the normal density for the dataset x
        mu is the mean of x
        var is variance  of x
    """
    num = np.exp(- ((x - mu) ** 2)/(2 * var))
    den = np.sqrt(2 * np.pi * var)
    return num / den


def likelihood(x, mu, var):
    """computes the likelihood for the dataset x
       mu is mean of x
       var is variance of x
    """
    ll_samples = GAU_pdf(x, mu, var)
    return ll_samples.prod()


def GAU_logpdf(x, mu, var):
    """ computes the log-density of the dataset x
        mu is the mean of x
        var is variance of x
    """
    first = -(1/2) * np.log(2 * np.pi)
    second = -(1/2) * np.log(var)
    third = -(((x - mu) ** 2) / (2 * var))
    return first + second + third


def log_likelihood(x, mu, var):
    """ computes the log-likelihood of the dataset x
        mu is the mean of x
        var is variance of x
    """
    log_ll_samples = GAU_logpdf(x, mu, var)
    log_likelihood = log_ll_samples.sum()
    return log_likelihood


def calculate_ML_estimates(x):
    """returns muML and varML for the dataset x """

    # those are the parameters that better describe the dataset x
    # they can be calculated in this way
    muML = XGAU.sum() / len(XGAU)
    # or alternatively
    # muML = XGAU.mean()
    varML = (((XGAU - muML) ** 2).sum())/len(XGAU)
    # or alternatively
    # varML = XGAU.var()
    return (muML, varML)


def logpdf_GAU_ND(x, mu, C):
    """Computes the Multivariate Gaussian log density for the dataset x
       C represents the covariance matrix sigma
    """
    # M is the number of rows of x, n of attributes for each sample
    M = x.shape[0]
    first = -(M/2) * np.log(2*np.pi)
    second = -0.5 * np.linalg.slogdet(C)[1]
    third = -0.5 * np.dot(
        np.dot((x-mu).T, np.linalg.inv(C)), (x - mu))
    return np.diag(first+second+third)


if __name__ == "__main__":
    # Load the data
    XGAU = np.load('XGau.npy')
    # Plot the normalized histogram of the dataset
    plt.figure()
    plt.hist(XGAU, bins=50, density=True)
    # plt.show()

    # Compute the normal density for XPlot density calling the function GAU_pdf
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot, GAU_pdf(XPlot, 1.0, 2.0))
    # plt.show()

    # Check that the density is correct, compare it with the solution
    pdfSol = np.load('CheckGAUPdf.npy')
    pdfGau = GAU_pdf(XPlot, 1.0, 2.0)
    # print(np.abs(pdfSol - pdfGau).mean())

    # Compute likelihood for our dataset XGAU, mu=0, var=1
    likel = likelihood(XGAU, 1.0, 2.0)
    # we get likelihood = 0.0, because of numerical issues
    # we can see that ll_samples are all small numbers, so if we
    # compute their product, it saturates to zero

    # For this reason we use log of density, rather than density
    # Compute log-likelihood for dataset XGAU, mu=0, var=1, it uses log density
    log_likel = log_likelihood(XGAU, 1.0, 2.0)

    # We want to estimate the parameters that better describe our dataset XGAU
    # Call the function that computes the ML(Maximum Likelihood)  estimates
    (muML, varML) = calculate_ML_estimates(XGAU)

    # Compute the log-likelihood using the ML estimates
    log_likel_GAU = log_likelihood(XGAU, muML, varML)
    # print(log_likel_GAU)

    # Plot the log_density computed using muML and varML of XPlot
    # on top of the histogram
    plt.figure()
    plt.hist(XGAU, bins=50, density=True)
    plt.plot(XPlot, np.exp(GAU_logpdf(XPlot, muML, varML)))
    # plt.show()

    # Multivariate Gaussian Density
    # Check if we implemented correctly logpdf_GAU_ND
    # Load the data
    XND = np.load('XND.npy')
    muND = np.load('muND.npy')
    CND = np.load('CND.npy')

    pdfSolND = np.load('llND.npy')
    pdfGauND = logpdf_GAU_ND(XND, muND, CND)

    print("SOL:\n")
    print(pdfSolND)
    print("\nMY RESULT: \n")
    print(pdfGauND)
