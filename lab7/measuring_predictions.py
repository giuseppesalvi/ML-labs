from modules import gaussian_models as gm

if __name__ == "__main__":
    # Load iris dataset
    D, L = gm.load_iris()

    # Split dataset in training set and test set
    (DTR, LTR), (DTE, LTE) = gm.split_db_2to1(D, L)

    # Multivariate Gaussian Classifier (MVG), implementation with log-densities
    gm.multivariate_gaussian_classifier2(DTR, LTR, DTE, LTE, print_flag=False)

