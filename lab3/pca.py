import numpy
import matplotlib.pyplot as plt
import scipy.linalg


def load(filename):
    """ Loads the dataset from the filename passed as argument
        Returns a matrix with the attributes of the samples
        And an array with the classes
    """
    with open(filename, 'r') as f:
        samples = []
        labels = []
        classes = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,
        }

        for line in f:
            if line[:-1] == "":  # skip empty lines
                continue
            s_l, s_w, p_l, p_w, label = line.split(",")
            a = numpy.array([float(s_l), float(s_w), float(p_l), float(p_w)])
            samples.append(a.reshape((4, 1)))
            # [:-1] is used to trim the newline
            labels.append(classes[label[:-1]])
        samples_matrix = numpy.hstack(samples)
        labels_array = numpy.array(labels)
        return samples_matrix, labels_array


def plot_2D_data(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]
    plt.figure()
    plt.xlabel("first principal direction")
    plt.ylabel("second principal direction")
    plt.scatter(D0[0], D0[1], label='Setosa')
    plt.scatter(D1[0], D1[1], label='Versicolor')
    plt.scatter(D2[0], D2[1], label='Virginica')

    plt.legend()
    plt.tight_layout()
    plt.show()


def covariance_matrix1(D):
    """ Computes and returns the covariance matrix given the dataset D
        Straightforward implemaentation based on for loops
        Not efficient
    """
    # compute the dataset mean mu
    mu = 0
    # D.shape[1] is the number of columns, D.shape[0] would be n of rows
    # D[:, i:i+1] is a column vector of shape (1, D.shape[1])
    # corresponding to the ith row of D
    for i in range(D.shape[1]):
        mu = mu + D[:, i:i+1]
    mu = mu / float(D.shape[1])
    # print(mu)
    # compute the covariance matrix
    C = 0
    for i in range(D.shape[1]):
        C = C + numpy.dot(D[:, i:i+1] - mu, (D[:, i:i+1] - mu).T)
    C = C / float(D.shape[1])
    # print(C)
    return C


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
    C = numpy.dot(DC, DC.T)
    C = C / float(D.shape[1])
    # print(C)
    return C


def leading_eigenvectors1(C, m):
    """ retrieves the m leading eigenvectors from the covariance matrix """
    # compute the eigevalues and eigenvectors
    s, U = numpy.linalg.eigh(C)
    # since s originally sort the eigenvalues from smallest to largest
    # we need to invert the order of the eigenvectors
    # and extract the m leading ones
    P = U[:, ::-1][:, 0:m]
    return P


def leading_eigenvectors2(C, m):
    """ retreieves the m leading eigenvectors form the covariance matrix
        in this implementation we use the fact that C is semi-definite positive
        and we can get the sorted eigenbvectors from the svd
    """
    # C = U Sigma Vtransposed
    U, s, Vh = numpy.linalg.svd(C)
    # in this case the singular values, which are equal to the eigenvalues are
    # already sorted in descending order, and the columns of U are the
    # corresponding eigenvectors
    P = U[:, 0:m]
    return P


def pca(D, m):
    # compute the covariance matrix of D
    C = covariance_matrix2(D)
    # alternative
    # P = covariance_matrix1(D)

    # retrieve the m leading eigenvectors
    P = leading_eigenvectors1(C, m)
    # alternative
    # P = leading_eigenvectors2(C, m)

    # apply the projection to a matrix of samples
    DP = numpy.dot(P.T, D)
    return DP


def within_class_covariance_matrix(D):
    """ computes the within class covariance matrix SW for the dataset D"""

    # select the samples of the i-th class
    # in the IRIS dataset classes are labeled as 0, 1, 2
    D1 = D[:, L == 0]
    D2 = D[:, L == 1]
    D3 = D[:, L == 2]

    # to compute the within class covariance matrix, we have to sum
    # the covariance matrices of each class
    C1 = covariance_matrix1(D1)
    C2 = covariance_matrix1(D2)
    C3 = covariance_matrix1(D3)
    SW = C1 + C2 + C3
    return SW


def between_class_covariance_matrix(D):
    """ computes the between class covariance matrix SB for the dataset D"""

    # select the samples of the i-th class
    # in the IRIS dataset classes are labeled as 0, 1, 2
    D1 = D[:, L == 0]
    D2 = D[:, L == 1]
    D3 = D[:, L == 2]

    # to compute the between class covariance matrix we use its definition
    mu = D.mean(1)
    mu = mu.reshape((mu.size, 1))
    mu1 = D1.mean(1)
    mu1 = mu1.reshape((mu1.size, 1))
    mu2 = D2.mean(1)
    mu2 = mu2.reshape((mu2.size, 1))
    mu3 = D3.mean(1)
    mu3 = mu3.reshape((mu3.size, 1))
    SB1 = numpy.dot((mu1 - mu), (mu1 - mu).T)
    SB2 = numpy.dot((mu2 - mu), (mu2 - mu).T)
    SB3 = numpy.dot((mu3 - mu), (mu3 - mu).T)
    SB = D1.shape[1] * SB1 + D2.shape[1] * SB2 + D3.shape[1] * SB3
    SB = SB / float(D.shape[1])
    return SB


def compute_SB_SW(D):
    """ computes the between and within class covariance matrices
        returns them in the order SB, SW
    """
    SB = between_class_covariance_matrix(D)
    SW = within_class_covariance_matrix(D)
    return SB, SW


def generalized_eigenvalue_problem(SB, SW, m):
    """ returns the LDA directions (columns of W)
        that can be computed solving the generalized eigenvalue problem
    """
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    # The columns of W are not necessariliy orthogonal
    # if we want we can find a basis U for the subspace spanned by W
    # using the singular value decomposition of W

    # UW, _, _ = numpy.linalg.svd(W)
    # U = UW[:, 0:m]
    return W


def joint_diagonalization_SB_SW(SB, SW, m):
    U, s, _ = numpy.linalg.svd(SW)
    s = s.reshape(1, s.size)
    P1 = numpy.dot(U * (1.0/(s**0.5)), U.T)
    SBT = numpy.dot(numpy.dot(P1, SB), P1.T)
    P2 = leading_eigenvectors2(SBT, m)
    W = numpy.dot(P1.T, P2)
    return W


def lda(D, m):
    SB, SW = compute_SB_SW(D)
    W = generalized_eigenvalue_problem(SB, SW, m)
    # alternative
    # W = joint_diagonalization_SB_SW(SB, SW, m)
    DL = numpy.dot(W.T, D)
    return DL


if __name__ == '__main__':
    # load the dataset into D, labels into L
    D, L = load("../lab2/iris.csv")
    # compute pca projection matrix on D for dimensionality m
    m = 2
    DP = pca(D, m)
    plot_2D_data(DP, L)
    DL = lda(D, m)
    plot_2D_data(DL, L)
