import numpy as np
from modules import gaussian_models as gm


def confusion_matrix(predicted_labels, real_labels, K):
    """ Computes the confusion matrix given the predicted labels and
        the real labels
        K is the size of the matrix (K x K)
    """
    # Initialize the matrix of size K x K with zeros
    conf_matrix = np.zeros((K, K))

    # The element of the matrix in position i,j represents the number
    # of samples belonging to class j that are predicted as class i
    for i in range(predicted_labels.size):
        conf_matrix[predicted_labels[i]][real_labels[i]] += 1

    # Print the confusion matrix
    print_confusion_matrix(conf_matrix, K)

    return conf_matrix


def print_confusion_matrix(confusion_matrix, n_classes):
    print("\t     Class")
    print("\t|  ", end='')
    for i in range(n_classes):
        print(i, "   ", end='')
    print("\n\t", end='')
    for i in range(n_classes):
        print("-----", end='')
    print("")
    for i in range(n_classes):
        if(i==0):
            print("Pred ", i, "| ", end='')
        else:
            print("     ", i, "| ", end='')
        for j in range(n_classes):
            print(confusion_matrix[i][j]," ", end='')
        print("")
    return


if __name__ == "__main__":
    # Load iris dataset
    D, L = gm.load_iris()

    # Split dataset in training set and test set
    (DTR, LTR), (DTE, LTE) = gm.split_db_2to1(D, L)

    # Multivariate Gaussian Classifier (MVG), implementation with log-densities
    predictions, _, _, _, _ = gm.multivariate_gaussian_classifier2(
        DTR, LTR, DTE, LTE, print_flag=False)

    # Compute the confusion matrix for the predictions of the MVG classifier
    # on the IRIS dataset
    print("\nConfusion matrix for the predictions of the MVG classifier\n")
    conf_mvg = confusion_matrix(predictions, LTE, 3)

    # Tied Covariance Gaussian Classifier
    predictions, _, _, _, _ = gm.tied_covariance_gaussian_classifier(DTR, LTR, DTE, LTE, print_flag=False)

    # Compute the confusion matrix for the predictions of the Tied covariance
    # classifier on the IRIS dataset
    print("\nConfusion matrix for the predictions of the Tied covariance classifier\n")
    conf_tied = confusion_matrix(predictions, LTE, 3)

