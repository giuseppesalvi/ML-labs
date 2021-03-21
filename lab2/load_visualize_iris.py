import numpy
import matplotlib.pyplot as plt


# Loads the dataset from the filename passed as argument
# Returns a matrix with the attributes of the samples
# And an array with the classes
def load(filename):
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


def plot_histogram(D, L):

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # let's filter only the data corrisponging to each class
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    attributes = {
        0: 'Sepal lenght',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Peral width'
    }

    for index in range(4):
        plt.figure()
        plt.xlabel(attributes[index])
        plt.hist(D0[index, :], bins=10, density=True, alpha=0.4,
                 label='Setosa')
        plt.hist(D1[index, :], bins=10, density=True, alpha=0.4,
                 label='Versicolor')
        plt.hist(D2[index, :], bins=10, density=True, alpha=0.4,
                 label='Virginica')
        plt.legend()
        plt.tight_layout()
    plt.show()


def plot_scatter(D, L):
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    attributes = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for index1 in range(4):
        for index2 in range(4):
            if index1 == index2:
                continue
            plt.figure()
            plt.xlabel(attributes[index1])
            plt.ylabel(attributes[index2])
            plt.scatter(D0[index1, :], D0[index2, :], label='Setosa')
            plt.scatter(D1[index1, :], D1[index2, :], label='Versicolor')
            plt.scatter(D2[index1, :], D2[index2, :], label='Virginica')
            plt.legend()
            plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    data, labels = load("iris.csv")
    plot_histogram(data, labels)
    plot_scatter(data, labels)
    mu = data.mean(1)
    centered_data = data - mu.reshape((data.shape[0], 1))
