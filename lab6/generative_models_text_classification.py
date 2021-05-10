import numpy as np
import scipy as sp


def load_data():
    with open('data/inferno.txt', 'r', encoding="ISO-8859-1") as f:
        inferno = []
        for line in f:
            inferno.append(line.strip())

    with open('data/purgatorio.txt', 'r', encoding="ISO-8859-1") as f:
        purgatorio = []
        for line in f:
            purgatorio.append(line.strip())

    with open('data/paradiso.txt', 'r', encoding="ISO-8859-1") as f:
        paradiso = []
        for line in f:
            paradiso.append(line.strip())

    return inferno, purgatorio, paradiso


def split_data(dataset, n):

    lTrain, lTest = [], []
    for i in range(len(dataset)):
        if i % n == 0:
            lTest.append(dataset[i])
        else:
            lTrain.append(dataset[i])

    return lTrain, lTest


def train_model(tercets, eps=0.1):
    """ Build a dictionary with inside a frequency dictionary for each class 
        tercets is a dictionary whose keys are the classes and values are the
        lists of tetcets of each class
        eps is epsilon, the smoothing factor
    """

    # Build dictionary of all possible words
    all_tercets = tercets['inferno'] + \
        tercets['purgatorio'] + tercets['paradiso']
    words_dict = build_dictionary(all_tercets)
    #
    # Build a dictionary for each class from the dictionary with all words
    # and initialize the values to eps
    inferno_dict = words_dict.fromkeys(words_dict, eps)
    purgatorio_dict = words_dict.fromkeys(words_dict, eps)
    paradiso_dict = words_dict.fromkeys(words_dict, eps)

    # Estimate counts for the tercets of each class
    # Initialize the number of total words with the sum of eps
    n_words_inferno = eps * len(inferno_dict)
    for tercet in tercets['inferno']:
        for word in tercet.split():
            inferno_dict[word] += 1
            n_words_inferno += 1

    n_words_purgatorio = eps * len(purgatorio_dict)
    for tercet in tercets['purgatorio']:
        for word in tercet.split():
            purgatorio_dict[word] += 1
            n_words_purgatorio += 1

    n_words_paradiso = eps * len(paradiso_dict)
    for tercet in tercets['paradiso']:
        for word in tercet.split():
            paradiso_dict[word] += 1
            n_words_paradiso += 1

    # Compute frequencies
    # In each class dictionary, for each key (word),
    # the value is the frequency of that word
    for word in inferno_dict:
        inferno_dict[word] = np.log(
            inferno_dict[word]) - np.log(n_words_inferno)
    for word in purgatorio_dict:
        purgatorio_dict[word] = np.log(
            purgatorio_dict[word]) - np.log(n_words_purgatorio)
    for word in paradiso_dict:
        paradiso_dict[word] = np.log(
            paradiso_dict[word]) - np.log(n_words_paradiso)

    # Return the three dictionaries inside a dictionary with labels as keys
    result = {'inferno': inferno_dict,
              'purgatorio': purgatorio_dict,
              'paradiso': inferno_dict}
    return result


def build_dictionary(tercets):
    """ Build a dictionary of all possible words contained in tercets
        the key is the word, the value is the number of occurrencies 
        tercets is a list of all the tercets
    """
    words_dict = {}
    for tercet in tercets:
        for word in tercet.split():
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1
    return words_dict


def compute_score_matrix(model, tercets):
    """ Computes the score matrix S for each class, given the tercets
        model is a dictionary conatining the model parameters, as returned
        by train_model() 
        tercets are the tercets for evaluation 
    """
    # Initialize a matrix of size N_classes * N_tercets with zeros
    # Each row corresponds to a class and each column to a test sample (tercet)
    S = np.zeros(len(model), len(tercets))
    for i, tercet in enumerate(tercets):
        scores = compute_log_likelihoods(model, tercet)
        # inferno: first row of the score matrix
        S[0][i] = scores['inferno']
        # purgatorio: second row of the score matrix
        S[1][i] = scores['purgatorio']
        # paradiso: third row of the score matrix
        S[2][i] = scores['paradiso']

    return S


def compute_log_likelihoods(model, text):
    '''
    Compute the array of log-likelihoods for each class for the given text
    model is the dictionary of model parameters as returned by train_model()
    The function returns a dictionary of class-conditional log-likelihoods
    '''
    log_likelihoods = {'inferno': 0, 'purgatorio': 0, 'paradiso': 0}

    # for each word in the text
    for word in text.split():
        if word in model['inferno']:
            log_likelihoods['inferno'] += model['inferno'][word]
        if word in model['purgatorio']:
            log_likelihoods['purgatorio'] += model['purgatorio'][word]
        if word in model['paradiso']:
            log_likelihoods['paradiso'] += model['paradiso'][word]

    return log_likelihoods


def mcol(v):
    return v.reshape((v.size, 1))


def compute_class_posteriors(S, log_prior=None):
    ''' Compute class posterior probabilities
        S: Score Matrix of class-conditional log-likelihoods
        log_prior: array with class prior probability 
        Returns: matrix of class posterior probabilities
    '''
    # If log_prior is non, uniform priors will be used
    if log_prior is None:
        logPrior = numpy.log(numpy.ones(S.shape[0]) / float(S.shape[0]))

    # Compute Joint probability
    SJoint = S + mcol(log_prior)

    # Compute the array of class posterior log_probabilities
    # Subtract marginal likelihood
    SPost = SJoint - sp.special.logsumexp(SJoint, axis=0)

    return np.exp(SPost)


if __name__ == "__main__":

    # Dataset

    # Load the three datasets
    inferno, purgatorio, paradiso = load_data()

    # Split datasets in train and test
    inferno_train, inferno_test = split_data(inferno, 4)
    purgatorio_train, purgatorio_test = split_data(purgatorio, 4)
    paradiso_train, paradiso_test = split_data(paradiso, 4)

    # Multinomial model for text

    # Approach: dictionaries of frequencies of each word in the text
    tercets_train = {'inferno': inferno_train,
                     'purgatorio': purgatorio_train,
                     'paradiso': paradiso_train}

    tercets_test = inferno_test + purgatorio_test + paradiso_test

    # Training the model
    model = train_model(tercets_train, eps=0.001)

    # Predicting the criteria

    # Score matrix
    S = compute_score_matrix(model, tercets_test)

    # Class posterior probabilities
    SPost = compute_class_posteriors(
        S, np.log(np.array([1./3., 1./3., 1./3.])))

    #TODO
    # labelsInf = numpy.zeros(len(lInfEval))
    # labelsInf[:] = hCls2Idx['inferno# 
    # labelsPar = numpy.zeros(len(lParEval))
    # labelsPar[:] = hCls2Idx['paradiso# 
    # labelsPur = numpy.zeros(len(lPurEval))
    # labelsPur[:] = hCls2Idx['purgatorio# 
    # labels_test= np.hstack([labelsInf, labelsPur, labelsPar])
    
    # 1 Task:
    # Closed-set multiclass classification:

    # The predicted label is obtained as the class that has maximum
    # posterior probability, (argmax is used for that
    predictions = SPost.argmax(axis=0)

    # Correct predictions, wrong predictions
    correct = (predictions.ravel() == L.ravel()).sum()
    wrong = predictions.size - correct
    
    # Accuracy, Error Rate
    accuracy = correct / predictions.size
    error_rate = wrong / predictions.size