import numpy
import matplotlib.pyplot as plt


def GAU_pdf(x, mu, var):
    """ computes the normal density for the dataset x
        mu is the mean of x
        var is variance  of x
    """
    num = numpy.exp(- ((x - mu) ** 2)/(2 * var))
    den = numpy.sqrt(2 * numpy.pi * var)
    return num / den


def GAU_logpdf(x, mu, var):
    """ computes the log-density of the dataset x
        mu is the mean of x
        var is variance of x
    """
    first = -(1/2) * numpy.log(2 * numpy.pi)
    second = -(1/2) * numpy.log(var)
    third = -(((x - mu) ** 2) / (2 * var))
    return first + second + third


def logpdf_GAU_ND(x, mu, C):
    # M is the number of rows of x, n of attributes for each sample
    M = x.shape[0]
    first = -(M/2) * numpy.log(2*numpy.pi)
    second = -0.5 * numpy.linalg.slogdet(C)[1]
    third = -0.5 * numpy.dot(numpy.dot((x-mu).T,
                             numpy.linalg.inv(C)), (x - mu))
    # we get the a matrix with the log densities of each sample in the diagonal
    return numpy.diag((first+second+third))


if __name__ == "__main__":
    # Load the data
    XGAU = numpy.load('XGau.npy')
    # Plot the normalized histogram of the dataset
    plt.figure()
    plt.hist(XGAU, bins=50, density=True)
    plt.show()

    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot, GAU_pdf(XPlot, 1.0, 2.0))
    plt.show()

    pdfSol = numpy.load('CheckGAUPdf.npy')
    pdfGau = GAU_pdf(XPlot, 1.0, 2.0)
    print(numpy.abs(pdfSol - pdfGau).mean())

    # Compute likelihood for our dataset XGAU, mu=0, var=1
    ll_samples = GAU_pdf(XGAU, 1.0, 2.0)
    likelihood = ll_samples.prod()
    print(likelihood)
    # we get likelihood = 0, because of numerical issues
    # we can see that ll_samples are all small numbers, so if we
    # compute their product, it saturates to zero

    # For this reason we use log of density, rather than density
    log_ll_samples = GAU_logpdf(XGAU, 1.0, 2.0)
    log_likelihood = log_ll_samples.sum()

    # Gaussian ML estimate
    muML = XGAU.sum() / len(XGAU)
    # muML2 = XGAU.mean()
    varML = (((XGAU - muML) ** 2).sum())/len(XGAU)
    # varML2 = XGAU.var() = varML

    # Compute the log-likelihood using the ML estimates
    # TODO create function loglikelighood(XGAU, m_ML, v_ML)
    log_ll_samples_GAU = GAU_logpdf(XGAU, muML, varML)
    ll_GAU = log_ll_samples_GAU.sum()
    print(ll_GAU)

    # Plot the log_density computed using muML and varML of XPlot
    # on top of the histogram
    plt.figure()
    plt.hist(XGAU, bins=50, density=True)
    plt.plot(XPlot, numpy.exp(GAU_logpdf(XPlot, muML, varML)))
    plt.show()

    # Multivariate Gaussian
    XND = numpy.load('XND.npy')
    muND = numpy.load('muND.npy')
    CND = numpy.load('CND.npy')
    pdfSolND = numpy.load('llND.npy')
    pdfGauND = logpdf_GAU_ND(XND, muND, CND)
    # chech the result
    print(numpy.abs(pdfSolND - pdfGauND).mean())
