import numpy as np
import sklearn.datasets as da
import scipy as sp


def load_iris():
    """ Load iris dataset from the sklearn library
        Returns the dataset matrix D and matrix of labels L
        We need to transpose the data matrix, since we work with column
        representations of feature vectors
    """
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


# FROM LAB 3
def covariance_matrix2(D):
    """ Computes and returns the covariance matrix given the dataset D
        this is a more efficient implementation
    """
    # compute the dataset mean mu
    mu = D.mean(1)
    # mu is a 1-D array, we need to reshape it to a column vector
    mu = mu.reshape((mu.size, 1))
    # print(mu)
    # remove the mean from all the points
    DC = D - mu
    # DC is the matrix of centered data
    C = np.dot(DC, DC.T)
    C = C / float(D.shape[1])
    # print(C)
    return C


def vcol(x):
    return x.reshape(x.shape[0], 1)


# FROM LAB 4
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


def multivariate_gaussian_classifier(DTR, LTR, DTE, LTE):
    """ implementation of the  Multivariate Gaussian Classifier
        DTR and LTR are training data and labels
        DTE and LTE are evaluation data and labels
    """
    # Compute the ML estimates for the classifier parameters
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    mu0 = DTR0.mean(1)
    C0 = covariance_matrix2(DTR0)
    mu1 = DTR1.mean(1)
    C1 = covariance_matrix2(DTR1)
    mu2 = DTR2.mean(1)
    C2 = covariance_matrix2(DTR2)

    # Compute for each test sample the likelihoods
    # Initialize the score matrix, S[i,j] contains the
    # class conditional probability for sample j given class i
    S = np.zeros((3, DTE.shape[1]))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[0, j] = np.exp(logpdf_GAU_ND(sample, vcol(mu0), C0))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[1, j] = np.exp(logpdf_GAU_ND(sample, vcol(mu1), C1))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[2, j] = np.exp(logpdf_GAU_ND(sample, vcol(mu2), C2))

    # Compute class posterior probabilities combining the score matrix with
    # prior information, in this case P(c) = 1/3 for every class
    SJoint = 1/3 * S
    SPost = SJoint / SJoint.sum(axis=0)

    # SPost is the array of class posterior probabilities
    # The predicted label is obtained as the class that has maximum
    # posterior probability, (argmax is used for that)
    predictions = SPost.argmax(axis=0)

    # Print Results
    print("MULTIVARIATE GAUSSIAN CLASSIFIER: ")

    # Print predicted labels, truth of the prediction, correct label if false
    print("\nPREDICTED LABELS:")
    for i, pred in enumerate(predictions):
        print("predicted: " + str(pred), pred == LTE[i],
              "" if pred == LTE[i] else "correct: " + str(LTE[i]))

    # correct predictions, wrong predictions
    predicted = (predictions == LTE).sum()
    not_predicted = predictions.size - predicted

    # accuracy, error rate
    acc = predicted / predictions.size
    err = not_predicted / predictions.size

    print("\nCORRECT PREDICTIONS : ", predicted)
    print("\nWRONG PREDICTIONS : ", not_predicted)
    print("\nACCURACY : ", acc)
    print("\nERROR RATE : ", err, "\n")


def multivariate_gaussian_classifier2(DTR, LTR, DTE, LTE):
    """ implementation of the  Multivariate Gaussian Classifier
        using log_densities
        DTR and LTR are training data and labels
        DTE and LTE are evaluation data and labels
    """
    # Compute the ML estimates for the classifier parameters
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    mu0 = DTR0.mean(1)
    C0 = covariance_matrix2(DTR0)
    mu1 = DTR1.mean(1)
    C1 = covariance_matrix2(DTR1)
    mu2 = DTR2.mean(1)
    C2 = covariance_matrix2(DTR2)

    # Compute for each test sample the likelihoods
    # Initialize the score matrix, S[i,j] contains the
    # class conditional probability for sample j given class i
    S = np.zeros((3, DTE.shape[1]))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[0, j] = logpdf_GAU_ND(sample, vcol(mu0), C0)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[1, j] = logpdf_GAU_ND(sample, vcol(mu1), C1)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[2, j] = logpdf_GAU_ND(sample, vcol(mu2), C2)

    # Compute class posterior probabilities combining the score matrix with
    # prior information, in this case P(c) = 1/3 for every class
    SJoint = np.log(1/3) + S
    SPost = SJoint - sp.special.logsumexp(SJoint, axis=0)

    # SPost is the array of class posterior probabilities
    # The predicted label is obtained as the class that has maximum
    # posterior probability, (argmax is used for that)
    predictions = SPost.argmax(axis=0)

    # Print Results
    print("MULTIVARIATE GAUSSIAN CLASSIFIER: ")
    print("(IMPLEMENTATION WITH LOG-DENSITIES)")

    # Print predicted labels, truth of the prediction, correct label if false
    print("\nPREDICTED LABELS:")
    for i, pred in enumerate(predictions):
        print("predicted: " + str(pred), pred == LTE[i],
              "" if pred == LTE[i] else "correct: " + str(LTE[i]))

    # correct predictions, wrong predictions
    predicted = (predictions == LTE).sum()
    not_predicted = predictions.size - predicted

    # accuracy, error rate
    acc = predicted / predictions.size
    err = not_predicted / predictions.size

    print("\nCORRECT PREDICTIONS : ", predicted)
    print("\nWRONG PREDICTIONS : ", not_predicted)
    print("\nACCURACY : ", acc)
    print("\nERROR RATE : ", err, "\n")


if __name__ == "__main__":

    # Load iris dataset
    D, L = load_iris()

    # Split dataset in training set and test set
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # Multivariate Gaussian Classifier (MVG)
    multivariate_gaussian_classifier(DTR, LTR, DTE, LTE)

    # Multivariate Gaussian Classifier (MVG), implementation with log-densities
    multivariate_gaussian_classifier2(DTR, LTR, DTE, LTE)
