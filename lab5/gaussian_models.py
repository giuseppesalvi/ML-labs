import numpy as np
import sklearn.datasets as da
import scipy as sp


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


# FROM LAB 3
def covariance_matrix2(D):
    """ Computes and returns the covariance matrix given the dataset D
        this is a more efficient implementation
    """
    # compute the dataset mean mu
    mu = D.mean(1)

    # mu is a 1-D array, we need to reshape it to a column vector
    mu = vcol(mu)

    # remove the mean from all the points
    DC = D - mu

    # DC is the matrix of centered data
    C = np.dot(DC, DC.T)
    C = C / float(D.shape[1])

    return C


def vcol(x):
    """ reshape the vector x into a column vector """

    return x.reshape(x.shape[0], 1)


# FROM LAB 4
def logpdf_GAU_ND(x, mu, C):
    """ Computes the Multivariate Gaussian log density for the dataset x
        C represents the covariance matrix sigma
    """
    # M is the number of rows of x, n of attributes for each sample
    M = x.shape[0]
    first = -(M/2) * np.log(2*np.pi)
    second = -0.5 * np.linalg.slogdet(C)[1]
    third = -0.5 * np.dot(
        np.dot((x-mu).T, np.linalg.inv(C)), (x - mu))

    return np.diag(first+second+third)


def multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=False):
    """ implementation of the  Multivariate Gaussian Classifier
        DTR and LTR are training data and labels
        DTE and LTE are evaluation data and labels
        print_flag = True to print results, false otherwise
        returns: the predicitons, the number of correct predictions,
            the number of wrong predictions, the accuracy and the error rate
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

    # correct predictions, wrong predictions
    predicted = (predictions == LTE).sum()
    not_predicted = predictions.size - predicted

    # accuracy, error rate
    acc = predicted / predictions.size
    err = not_predicted / predictions.size

    # Print Results
    if(print_flag):
        t = "MULTIVARIATE GAUSSIAN CLASSIFIER: \n"
        print_results(t, predictions, LTE, predicted, not_predicted, acc, err)

    return predictions, predicted, not_predicted, acc, err


def multivariate_gaussian_classifier2(DTR, LTR, DTE, LTE, print_flag=False):
    """ implementation of the  Multivariate Gaussian Classifier
        using log_densities
        DTR and LTR are training data and labels
        DTE and LTE are evaluation data and labels
        print_flag = True to print results, false otherwise
        returns: the predicitons, the number of correct predictions,
            the number of wrong predictions, the accuracy and the error rate
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

    # correct predictions, wrong predictions
    predicted = (predictions == LTE).sum()
    not_predicted = predictions.size - predicted

    # accuracy, error rate
    acc = predicted / predictions.size
    err = not_predicted / predictions.size

    # Print Results
    if(print_flag):
        t = "MULTIVARIATE GAUSSIAN CLASSIFIER: \n" + \
            "(IMPLEMENTATION WITH LOG-DENSITIES)"
        print_results(t, predictions, LTE, predicted, not_predicted, acc, err)

    return predictions, predicted, not_predicted, acc, err


def print_results(title, predictions, LTE, n_correct, n_wrong, acc, err):
    """ Prints the predicted labels, the number of correct predictions,
        the number of wrong predictions, the accuracy and the error rate
    """
    # Print Title
    print("-----------------------------------------------------------")
    print(title)
    print("-----------------------------------------------------------")

    # Print predicted labels, truth of the prediction, correct label if false
    print("PREDICTED LABELS:")
    for i, pred in enumerate(predictions):
        print("predicted: " + str(pred), pred == LTE[i],
              "" if pred == LTE[i] else "correct: " + str(LTE[i]))

    # Print number of correct and wrong predictions
    print("\nCORRECT PREDICTIONS : ", n_correct, "\n")
    print("WRONG PREDICTIONS : ", n_wrong, "\n")

    # Print accuracy and error rate
    print("ACCURACY : ", acc,  "\n")
    print("ERROR RATE : ", err)
    print("-----------------------------------------------------------\n\n")

    return


def naive_bayes_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=False):
    """ implementation of the  Naive Bayes Gaussian Classifier
        based on MVG version with log_densities,
        covariance matrixes are diagonal
        DTR and LTR are training data and labels
        DTE and LTE are evaluation data and labels
        print_flag = True to print results, false otherwise
        returns: the predicitons, the number of correct predictions,
            the number of wrong predictions, the accuracy and the error rate
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

    # We need to zeroing the out of diagonal elements of the MVG ML solution
    # This can be done multiplying element-wise the MVG ML solution
    # with the identity matrix
    C0 *= np.identity(C0.shape[0])
    C1 *= np.identity(C1.shape[0])
    C2 *= np.identity(C2.shape[0])

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

    # correct predictions, wrong predictions
    predicted = (predictions == LTE).sum()
    not_predicted = predictions.size - predicted

    # accuracy, error rate
    acc = predicted / predictions.size
    err = not_predicted / predictions.size

    # Print Results
    if(print_flag):
        t = "NAIVE BAYES GAUSSIAN CLASSIFIER"
        print_results(t, predictions, LTE, predicted, not_predicted, acc, err)

    return predictions, predicted, not_predicted, acc, err


# FROM LAB 3
def within_class_covariance_matrix(DTR, LTR):
    """ computes the within class covariance matrix SW for the dataset D"""

    # select the samples of the i-th class
    # in the IRIS dataset classes are labeled as 0, 1, 2
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    # to compute the within class covariance matrix, we have to sum
    # the covariance matrices of each class
    C0 = covariance_matrix2(DTR0)
    C1 = covariance_matrix2(DTR1)
    C2 = covariance_matrix2(DTR2)
    SW = (DTR0.shape[1] * C0 +
          DTR1.shape[1] * C1 +
          DTR2.shape[1] * C2) / DTR.shape[1]
    return SW


def tied_covariance_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=False):
    """ implementation of the Tied Covariance Gaussian Classifier
        based on MVG version with log_densities
        DTR and LTR are training data and labels
        DTE and LTE are evaluation data and labels
        print_flag = True to print results, false otherwise
        returns: the predicitons, the number of correct predictions,
            the number of wrong predictions, the accuracy and the error rate
    """
    # Compute the ML estimates for the classifier parameters
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]

    mu0 = DTR0.mean(1)
    mu1 = DTR1.mean(1)
    mu2 = DTR2.mean(1)

    C_star = within_class_covariance_matrix(DTR, LTR)

    # Compute for each test sample the likelihoods
    # Initialize the score matrix, S[i,j] contains the
    # class conditional probability for sample j given class i
    S = np.zeros((3, DTE.shape[1]))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[0, j] = logpdf_GAU_ND(sample, vcol(mu0), C_star)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[1, j] = logpdf_GAU_ND(sample, vcol(mu1), C_star)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[2, j] = logpdf_GAU_ND(sample, vcol(mu2), C_star)

    # Compute class posterior probabilities combining the score matrix with
    # prior information, in this case P(c) = 1/3 for every class
    SJoint = np.log(1/3) + S
    SPost = SJoint - sp.special.logsumexp(SJoint, axis=0)

    # SPost is the array of class posterior probabilities
    # The predicted label is obtained as the class that has maximum
    # posterior probability, (argmax is used for that)
    predictions = SPost.argmax(axis=0)

    # correct predictions, wrong predictions
    predicted = (predictions == LTE).sum()
    not_predicted = predictions.size - predicted

    # accuracy, error rate
    acc = predicted / predictions.size
    err = not_predicted / predictions.size

    # Print Results
    if(print_flag):
        t = "TIED COVARIANCE GAUSSIAN CLASSIFIER: "
        print_results(t, predictions, LTE, predicted, not_predicted, acc, err)

    return predictions, predicted, not_predicted, acc, err


def k_fold(D, L, K, seed=0):
    """ implementation of the k-fold cross validation approach
        D is the dataset, L the labels, K the number of folds
        it prints out the results
    """
    sizePartitions = int(D.shape[1]/K)
    np.random.seed(seed)

    # permutate the indexes of the samples
    idx_permutation = np.random.permutation(D.shape[1])

    # put the indexes inside different partitions
    idx_partitions = []
    for i in range(0, D.shape[1], sizePartitions):
        idx_partitions.append(list(idx_permutation[i:i+sizePartitions]))
    error_rates = {'MVG': 0.0, 'NAIVE': 0.0, 'TIED': 0.0}
    accuracies = {'MVG': 0.0, 'NAIVE': 0.0, 'TIED': 0.0}

    # for each fold, consider the ith partition in the test set
    # the other partitions in the train set
    for i in range(K):
        # keep the i-th partition for test
        # keep the other partitions for train
        idx_test = idx_partitions[i]
        idx_train = idx_partitions[0:i] + idx_partitions[i+1:]

        # from lists of lists collapse the elemnts in a single list
        idx_train = sum(idx_train, [])

        # partition the data and labels using the already partitioned indexes
        DTR = D[:, idx_train]
        DTE = D[:, idx_test]
        LTR = L[idx_train]
        LTE = L[idx_test]

        # Multivariate Gaussian Classifier
        _, _, _,\
            acc, err = multivariate_gaussian_classifier2(DTR, LTR, DTE, LTE)
        accuracies['MVG'] += acc / K
        error_rates['MVG'] += err / K

        # Naive Bayes Gaussian Classifier
        _, _, _,\
            acc, err = naive_bayes_gaussian_classifier(DTR, LTR, DTE, LTE)
        accuracies['NAIVE'] += acc / K
        error_rates['NAIVE'] += err / K

        # Tied Covariance Gaussian Classifier
        _, _, _,\
            acc, err = tied_covariance_gaussian_classifier(DTR, LTR, DTE, LTE)
        accuracies['TIED'] += acc / K
        error_rates['TIED'] += err / K

    print("-----------------------------------------------------------")
    print("K-FOLD Cross Validation, K =", K)
    print("-----------------------------------------------------------")
    print("RESULTS:")
    print("Multivariate Gaussian Classifier",
          "accuracy: ", accuracies['MVG'] * 100, "% ",
          "error rate:", error_rates['MVG'] * 100, "%")
    print("Naive Bayes Gaussian Classifier",
          "accuracy: ", accuracies['NAIVE'] * 100, "% ",
          "error rate:", error_rates['NAIVE'] * 100, "%")
    print("Tied Covariance Gaussian Classifier",
          "accuracy: ", accuracies['TIED'] * 100, "% ",
          "error rate:", error_rates['TIED'] * 100, "%")
    print("-----------------------------------------------------------\n\n")

    return


if __name__ == "__main__":

    # Load iris dataset
    D, L = load_iris()

    # Split dataset in training set and test set
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # Multivariate Gaussian Classifier (MVG)
    multivariate_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=True)

    # Multivariate Gaussian Classifier (MVG), implementation with log-densities
    multivariate_gaussian_classifier2(DTR, LTR, DTE, LTE, print_flag=True)

    # Naive Bayes Gaussian Classifier
    naive_bayes_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=True)

    # Tied Covariance Gaussian Classifier
    tied_covariance_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=True)

    # K-Fold
    k_fold(D, L, 150)
