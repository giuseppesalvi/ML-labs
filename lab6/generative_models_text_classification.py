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


if __name__ == "__main__":

    # Dataset

    # Load the three datasets
    inferno, purgatorio, paradiso = load_data()

    # Split datasets in train and test
    inferno_train, inferno_test = split_data(inferno, 4)
    purgatorio_train, purgatiorio_test = split_data(purgatorio, 4)
    paradiso_train, paradiso_test = split_data(paradiso, 4)
