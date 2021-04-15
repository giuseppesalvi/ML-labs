import numpy as np
import sklearn.datasets as da


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


if __name__ == "__main__":

    # Load iris dataset
    D, L = load_iris()

    # Split dataset in training set and test set
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    mu0 = DTR0.mean(1)
    C0 = covariance_matrix2(DTR0)
    mu1 = DTR1.mean(1)
    C1 = covariance_matrix2(DTR1)
    mu2 = DTR2.mean(1)
    C2 = covariance_matrix2(DTR2)

