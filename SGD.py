import pickle
import numpy as np
import matplotlib.pyplot as plt

def loadData(file):
    with open(file, 'rb') as f: 
        training_data, validation_data, test_data = pickle.load(f, encoding='bytes')

    test_data       = list(zip(test_data[0], test_data[1]))
    training_data   = list(zip(training_data[0], training_data[1]))
    validation_data = list(zip(validation_data[0], validation_data[1]))
    
    return training_data, validation_data, test_data

class Network(object):

    def __init__(self, nn):
        self.size  = len(nn)
        self.shape = nn

        self.weights = []
        self.biases  = []
        for i in range(self.size - 1):
            self.weights.append(np.random.normal(0, 1/np.sqrt(nn[i]), (nn[i+1], nn[i])))
            self.biases.append(np.random.normal(0, 1, (nn[i+1], 1)))

    def activation(self, z):
        return 1/(1 + np.exp(-z))
    def activation_d(self, z):
        return np.exp(-z)/(1 + np.exp(-z))/(1 + np.exp(-z))

    def costFunction(self, target):
        return np.linalg.norm(self.a[-1] - target)^2/2
    def costFunction_d(self, output, target):
        return output - target

    def feedForward(self, input):
        self.z = []
        self.a = [input]
        for i in range(self.size - 1):
            self.z.append(np.dot(self.weights[i], self.a[i]) + self.biases[i])
            self.a.append(self.activation(self.z[i]))
            
    def backPropagation(self, target, eta):
        self.delta     = [None] * (self.size - 1)
        self.delta[-1] = np.array([[self.costFunction_d(self.a[-1][i][0], target[i][0])] for i in range(len(self.a[-1]))])

        for i in range(2, self.size):
            self.delta[-i] = np.dot(np.transpose(self.weights[-i+1]), self.delta[-i+1]) * self.activation_d(self.z[-i])

        for i in range(1, self.size):
            self.weights[-i] = self.weights[-i] - eta*np.dot(self.delta[-i], np.transpose(self.a[-i-1]))
            self.biases[-i]  = self.biases[-i] - eta*self.delta[-i]

    def miniBatches(self, trainig_data, mini_batch_size):
        np.random.shuffle(trainig_data)
        n            = int(np.ceil(len(trainig_data)/mini_batch_size))
        mini_batches = [trainig_data[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(n)]

        return mini_batches

    def SGD(self, data_in, eta, epochs, mini_batch_size):
        mini_batches = self.miniBatches(data_in, mini_batch_size)

        for _ in range(epochs):
            for mini_batch in mini_batches:
                for digit in mini_batch:
                    input            = digit[0].reshape((-1,1))
                    target           = np.zeros((10,1))
                    target[digit[1]] = 1
                    self.feedForward(input)
                    self.backPropagation(target, eta)

        print("done : {} epochs".format(epochs))


if __name__ == "__main__":

    training_data, validation_data, test_data = loadData('data/mnist.pkl')

    # plt.imshow(training_data[0][1020].reshape((28,28)))
    # plt.show()
    # print(training_data[1][1020])

    nn = Network([784,20,10])
    nn.SGD(training_data, eta=0.1, epochs=1, mini_batch_size=20)

    mini_batches = nn.miniBatches(test_data, 20)
    input        = mini_batches[0][0][0].reshape((-1,1))
    target       = mini_batches[0][0][1]
    nn.feedForward(input)
    print(target, np.argwhere(nn.a[-1]==np.max(nn.a[-1])))
