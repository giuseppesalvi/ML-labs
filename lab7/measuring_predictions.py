import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import log
from modules import gaussian_models as gm
from modules import generative_models_text_classification as gmtx


def confusion_matrix(predicted_labels, real_labels, K, print_flag=True):
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
    if print_flag:
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
        if(i == 0):
            print("Pred ", i, "| ", end='')
        else:
            print("     ", i, "| ", end='')
        for j in range(n_classes):
            print(confusion_matrix[i][j], " ", end='')
        print("")
    return


def optimal_bayes_decisions(llr, pi1, Cfn, Cfp, threshold=None):
    """ Computes optimal Bayes decisions starting from the binary 
        log-likelihoods ratios
        llr is the array of log-likelihoods ratios
        pi1 is the prior class probability of class 1 (True)
        Cfp = C1,0 is the cost of false positive errors, that is the cost of 
        predicting class 1 (True) when the actual class is 0 (False)
        Cfn = C0,1 is the cost of false negative errors that is the cost of 
        predicting class 0 (False) when the actual class is 1 (True)
    """

    # initialize an empty array for predictions of samples
    predictions = np.empty(llr.shape, int)

    # compare the log-likelihood ratio with threshold to predict the class
    # if the threshold is not specified use the theoretical optimal threshold
    if (threshold == None):
        threshold = - log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    for i in range(llr.size):
        if llr[i] > threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions


def empirical_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
    """ Computes the Bayes risk (or detection cost) from the consufion matrix 
        corresponding to the optimal decisions for an 
        application (pi1, Cfn, Cfp)
    """

    # FNR = false negative rate
    FNR = confusion_matrix[0][1] / \
        (confusion_matrix[0][1] + confusion_matrix[1][1])

    # FPR = false positive rate
    FPR = confusion_matrix[1][0] / \
        (confusion_matrix[0][0] + confusion_matrix[1][0])

    # We can compute the empirical bayes risk, or detection cost function DCF
    # using this formula
    DCF = pi1 * Cfn * FNR + (1-pi1) * Cfp * FPR

    return DCF


def normalized_detection_cost(DCF, pi1, Cfn, Cfp):
    """ Computes the normalized detection cost, given the detection cost DCF,
        and the parameters of the application, pi1, Cfn, Cfp
    """

    # We can compute the normalized detection cost (or bayes risk)
    # by dividing the bayes risk by the risk of an optimal system that doen not
    # use the test data at all

    # The cost of such system is given by this formula
    DCFdummy = pi1 * Cfn if (pi1 * Cfn < (1-pi1) * Cfp) else (1-pi1) * Cfp

    return DCF / DCFdummy


def minimum_detection_costs(llr, labels, pi1, Cfn, Cfp):
    """ Compute the minimum detection cost, given the binary
        log likelihood ratios llr
        labels is the array of labels
        pi1, Cfn, Cfp are the parameters for the application
    """

    # consider a set of thresholds corresponding to (-inf, s1, ... , sM, inf)
    # where s1 ... sM are the test scores, sorted in increasing order
    thresholds = np.append(llr, [np.inf, -np.inf])
    thresholds.sort()

    min_DCF = np.inf
    for t in thresholds:
        # compare the log-likelihood ratio with threshold to predict the class
        predictions = optimal_bayes_decisions(llr, pi1, Cfn, Cfp, t)

        # compute the confusion matrix
        conf = confusion_matrix(predictions, labels, 2, False)

        # compute DCF_norm
        DCF = empirical_bayes_risk(conf, pi1, Cfn, Cfp)
        DCF_norm = normalized_detection_cost(DCF, pi1, Cfn, Cfp)

        # set DCF as minimum if true
        if DCF_norm < min_DCF:
            min_DCF = DCF_norm

    return min_DCF


def plot_ROC(llr, labels):
    """ plot the ROC curve TPR against FPR, given log likelihood ratios
        and labels 
    """

    # consider a set of thresholds corresponding to (-inf, s1, ... , sM, inf)
    # where s1 ... sM are the test scores, sorted in increasing order
    thresholds = np.append(llr, [np.inf, -np.inf])
    thresholds.sort()

    # inititialize empty vectors that will contain coordinates of points to plot
    FPRs = np.empty(thresholds.shape)
    TPRs = np.empty(thresholds.shape)

    for j, t in enumerate(thresholds):
        # initialize an empty array for predictions of samples
        predictions = np.empty(llr.shape, int)

        # compare the log-likelihood ratio with threshold to predict the class
        for i in range(llr.size):
            if llr[i] > t:
                predictions[i] = 1
            else:
                predictions[i] = 0

        # compute the confusion matrix
        conf = confusion_matrix(predictions, labels, 2, False)

        # FNR = false negative rate
        FNR = conf[0][1] / \
            (conf[0][1] + conf[1][1])

        # TPR = True positive rate
        TPR = 1 - FNR

        # FPR = false positive rate
        FPR = conf[1][0] / \
            (conf[0][0] + conf[1][0])

        # Add results to the arrays
        FPRs[j] = FPR
        TPRs[j] = TPR

    # Plot the ROC, on x-axis all the FPRs, on y-asis all the TPRs
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(FPRs, TPRs)
    plt.show()
    return


def bayes_error_plots(llr, labels, eff_prior_log_odds):
    """ Plot the normalized costs as a fuction of an effective prior pi_tilde """
    DCFs = np.empty(eff_prior_log_odds.size)
    min_DCFs = np.empty(eff_prior_log_odds.size)

    for j, p in enumerate(eff_prior_log_odds):
        pi_tilde = 1 / (1 + np.exp(-p))
        predictions_bayes = optimal_bayes_decisions(llr, pi_tilde, 1, 1, -p)
        # equivalent: predictions_bayes = optimal_bayes_decisions(llr, pi_tilde, 1, 1)
        conf = confusion_matrix(predictions_bayes, labels, 2, False)
        DCF = empirical_bayes_risk(conf, pi_tilde, 1, 1)
        DCF_norm = normalized_detection_cost(DCF, pi_tilde, 1, 1)
        DCF_min = minimum_detection_costs(llr, labels, pi_tilde, 1, 1)
        DCFs[j] = DCF_norm
        min_DCFs[j] = DCF_min

    plt.figure()
    plt.plot(eff_prior_log_odds, DCFs, label="DCF", color="r")
    plt.plot(eff_prior_log_odds, min_DCFs, label="min DCF", color="b")
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()
    return DCFs, min_DCFs


if __name__ == "__main__":

    # 1. Confusion matrices and accuracy

    # IRIS Dataset

    # Load iris dataset
    D, L = gm.load_iris()

    # Split dataset in training set and test set
    (DTR, LTR), (DTE, LTE) = gm.split_db_2to1(D, L)

    # Multivariate Gaussian Classifier (MVG), implementation with log-densities
    predictions_mvg, _, _, _, _ = gm.multivariate_gaussian_classifier2(
        DTR, LTR, DTE, LTE, print_flag=False)

    # Compute the confusion matrix for the predictions of the MVG classifier
    # on the IRIS dataset
    print("\nConfusion matrix for the predictions of the MVG classifier\n")
    conf_mvg = confusion_matrix(predictions_mvg, LTE, 3)

    # Tied Covariance Gaussian Classifier
    predictions_tied, _, _, _, _ = gm.tied_covariance_gaussian_classifier(
        DTR, LTR, DTE, LTE, print_flag=False)

    # Compute the confusion matrix for the predictions of the Tied covariance
    # classifier on the IRIS dataset
    print("\nConfusion matrix for the predictions of the Tied covariance classifier\n")
    conf_tied = confusion_matrix(predictions_tied, LTE, 3)

    # Divina commedia

    # Load the class-conditional likelihoods
    likelihoods_commedia = np.load("data/commedia_ll.npy")

    # Load the corresponding tercet labels
    labels_commedia = np.load("data/commedia_labels.npy")

    # Compute class posterior probabilities
    post_prob_commedia = gmtx.compute_class_posteriors(
        likelihoods_commedia, np.log(np.array([1./3., 1./3., 1./3.])))

    # The predicted label is obtained as the class that has maximum
    # posterior probability
    predictions_commedia = np.argmax(post_prob_commedia, axis=0)

    # Compute the confusion matrix for the predictions on Divina Commedia
    print("\nConfusion matrix for the predictions on divina commedia\n")
    conf_commedia = confusion_matrix(predictions_commedia, labels_commedia, 3)

    # 2. Binary taks: optimal Bayes decisions

    # Load log-likelihood ratios for the inferno-vs-paradiso task
    llr_infpar = np.load("data/commedia_llr_infpar.npy")

    # Load the corresponding labels
    labels_infpar = np.load("data/commedia_labels_infpar.npy")

    # Compute optimal Bayes decisions for the binary task inferno-vs-paradiso
    # And print the confusion matrix for different parameters
    print("\nConfusion matrix for the predictions using optimal Bayes decisions\n")

    # pi1 = 0.5, Cfn = 1, Cfp = 1
    predictions_bayes1 = optimal_bayes_decisions(llr_infpar, 0.5, 1, 1)
    print("\npi1 = 0.5, Cfn = 1, Cfp = 1\n")
    conf_bayes1 = confusion_matrix(predictions_bayes1, labels_infpar, 2)

    # pi1 = 0.8, Cfn = 1, Cfp = 1
    predictions_bayes2 = optimal_bayes_decisions(llr_infpar, 0.8, 1, 1)
    print("\npi1 = 0.8, Cfn = 1, Cfp = 1\n")
    conf_bayes2 = confusion_matrix(predictions_bayes2, labels_infpar, 2)

    # pi1 = 0.5, Cfn = 10, Cfp = 1
    predictions_bayes3 = optimal_bayes_decisions(llr_infpar, 0.5, 10, 1)
    print("\npi1 = 0.5, Cfn = 10, Cfp = 1\n")
    conf_bayes3 = confusion_matrix(predictions_bayes3, labels_infpar, 2)

    # pi1 = 0.8, Cfn = 1, Cfp = 10
    predictions_bayes4 = optimal_bayes_decisions(llr_infpar, 0.8, 1, 10)
    print("\npi1 = 0.8, Cfn = 1, Cfp = 10\n")
    conf_bayes4 = confusion_matrix(predictions_bayes4, labels_infpar, 2)

    # 3. Binary task: evaluation

    DCF1 = empirical_bayes_risk(conf_bayes1, 0.5, 1, 1)
    DCF2 = empirical_bayes_risk(conf_bayes2, 0.8, 1, 1)
    DCF3 = empirical_bayes_risk(conf_bayes3, 0.5, 10, 1)
    DCF4 = empirical_bayes_risk(conf_bayes4, 0.8, 1, 10)

    print("\nEmpirical bayes risk\n")
    print("(pi1, Cfn, Cfp)\tDCFu (B)")
    print("-------------------------")
    print("(0.5, 1, 1)\t%.3f" % (DCF1))
    print("(0.8, 1, 1)\t%.3f" % (DCF2))
    print("(0.5, 10, 1)\t%.3f" % (DCF3))
    print("(0.8, 1, 10)\t%.3f" % (DCF4))

    DCF1_norm = normalized_detection_cost(DCF1, 0.5, 1, 1)
    DCF2_norm = normalized_detection_cost(DCF2, 0.8, 1, 1)
    DCF3_norm = normalized_detection_cost(DCF3, 0.5, 10, 1)
    DCF4_norm = normalized_detection_cost(DCF4, 0.8, 1, 10)

    print("\nNormalized DCF\n")
    print("(pi1, Cfn, Cfp)\tDCF")
    print("-------------------------")
    print("(0.5, 1, 1)\t%.3f" % (DCF1_norm))
    print("(0.8, 1, 1)\t%.3f" % (DCF2_norm))
    print("(0.5, 10, 1)\t%.3f" % (DCF3_norm))
    print("(0.8, 1, 10)\t%.3f" % (DCF4_norm))

    # 4. Minimum detection costs

    DCF1_min = minimum_detection_costs(llr_infpar, labels_infpar, 0.5, 1, 1)
    DCF2_min = minimum_detection_costs(llr_infpar, labels_infpar, 0.8, 1, 1)
    DCF3_min = minimum_detection_costs(llr_infpar, labels_infpar, 0.5, 10, 1)
    DCF4_min = minimum_detection_costs(llr_infpar, labels_infpar, 0.8, 1, 10)

    print("\nMinimum detection Costs\n")
    print("(pi1, Cfn, Cfp)\tmin DCF")
    print("-------------------------")
    print("(0.5, 1, 1)\t%.3f" % (DCF1_min))
    print("(0.8, 1, 1)\t%.3f" % (DCF2_min))
    print("(0.5, 10, 1)\t%.3f" % (DCF3_min))
    print("(0.8, 1, 10)\t%.3f" % (DCF4_min))

    # 5. ROC curves
    plot_ROC(llr_infpar, labels_infpar)

    # 6. Bayes error plots
    eff_prior_log_odds = np.linspace(-3, 3, 21)
    DCFs, min_DCFs = bayes_error_plots(llr_infpar, labels_infpar, eff_prior_log_odds)

    # 7. Comparing recognizers
    llr_infpar_eps1 = np.load("data/commedia_llr_infpar_eps1.npy")
    labels_infpar_eps1 = np.load("data/commedia_labels_infpar_eps1.npy")

    DCF1_norm_eps1 = normalized_detection_cost(empirical_bayes_risk(confusion_matrix(optimal_bayes_decisions(llr_infpar_eps1, 0.5, 1, 1), labels_infpar_eps1, 2, False), 0.5, 1, 1), 0.5, 1, 1)
    DCF2_norm_eps1 = normalized_detection_cost(empirical_bayes_risk(confusion_matrix(optimal_bayes_decisions(llr_infpar_eps1, 0.8, 1, 1), labels_infpar_eps1, 2, False), 0.8, 1, 1), 0.8, 1, 1)
    DCF3_norm_eps1 = normalized_detection_cost(empirical_bayes_risk(confusion_matrix(optimal_bayes_decisions(llr_infpar_eps1, 0.5, 10, 1), labels_infpar_eps1, 2, False), 0.5, 10, 1), 0.5, 10, 1)
    DCF4_norm_eps1 = normalized_detection_cost(empirical_bayes_risk(confusion_matrix(optimal_bayes_decisions(llr_infpar_eps1, 0.8, 1, 10), labels_infpar_eps1, 2, False), 0.8, 1, 10), 0.8, 1, 10)

    print("\nNormalized DCF e = 1\n")
    print("(pi1, Cfn, Cfp)\tDCF")
    print("-------------------------")
    print("(0.5, 1, 1)\t%.3f" % (DCF1_norm_eps1))
    print("(0.8, 1, 1)\t%.3f" % (DCF2_norm_eps1))
    print("(0.5, 10, 1)\t%.3f" % (DCF3_norm_eps1))
    print("(0.8, 1, 10)\t%.3f" % (DCF4_norm_eps1))


    DCF1_min_eps1 = minimum_detection_costs(llr_infpar_eps1, labels_infpar_eps1, 0.5, 1, 1)
    DCF2_min_eps1 = minimum_detection_costs(llr_infpar_eps1, labels_infpar_eps1, 0.8, 1, 1)
    DCF3_min_eps1 = minimum_detection_costs(llr_infpar_eps1, labels_infpar_eps1, 0.5, 10, 1)
    DCF4_min_eps1 = minimum_detection_costs(llr_infpar_eps1, labels_infpar_eps1, 0.8, 1, 10)

    print("\nMinimum detection Costs e = 1\n")
    print("(pi1, Cfn, Cfp)\tmin DCF")
    print("-------------------------")
    print("(0.5, 1, 1)\t%.3f" % (DCF1_min_eps1))
    print("(0.8, 1, 1)\t%.3f" % (DCF2_min_eps1))
    print("(0.5, 10, 1)\t%.3f" % (DCF3_min_eps1))
    print("(0.8, 1, 10)\t%.3f" % (DCF4_min_eps1))

    DCFs_eps1, min_DCFs_eps1 = bayes_error_plots(llr_infpar_eps1, labels_infpar_eps1, eff_prior_log_odds)

    plt.figure()
    plt.plot(eff_prior_log_odds, DCFs, color="r", label= "DCF (e = 0.001)")
    plt.plot(eff_prior_log_odds, min_DCFs, color="b", label="min DCF (e = 0.001)")
    plt.plot(eff_prior_log_odds, DCFs_eps1, color="y", label="DCF (e = 1)")
    plt.plot(eff_prior_log_odds, min_DCFs_eps1, color="g", label="DCF (e = 1)")
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()
