import numpy as np


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
    model = build_dictionary(all_tercets)
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
    return {'inferno': inferno_dict, 'purgatorio': purgatorio_dict, 'paradiso': inferno_dict}


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
