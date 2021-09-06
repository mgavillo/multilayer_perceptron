import numpy as np
import pandas as pd
import argparse
import math
my_dict = {'M': 1, "B": 0}

class Activations:
    def __init__(self):
        pass

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.b = []
        self.phi = []
        self.mu = []
        self.eta = 0.5 
        self.epochs = 10000
    
    def add(self, layer):
        self.layers.append(layer)
    

    def calculate_weights_bias(self):
        for i in range(0, len(self.layers)):
            if i != 0:
                self.layers[i].weights = np.random.rand(self.layers[i].n_neurons, self.layers[i - 1].n_neurons)
                self.layers[i].bias = np.ones(self.layers[i].n_neurons)
                # self.layers[i].bias = np.random.rand(self.layers[i].n_neurons)

    def sigmoid(self, X):
        z=np.array(X,dtype=np.float32)
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, X):
        s = self.sigmoid(X)
        return s*(1-s)

    def gradient_error(self, Y_train):
        Y_train = np.array([my_dict[value] for value in Y_train])
        # print("\n")
        # print(Y_train)

        # print(Y_train.shape)
        return Y_train - self.layers[3].neurons

    # def update_weights(self):
    #     for i in range(0, len(self.layers)):
    #         if i != 0:
    #             self.layers[i].weights = np.dot(sigmoid(self.layers[i - 1].neurons).T, )

        # w = np.ones(X.shape[1])
        #     for _ in tqdm(range(self.n_iter), desc=house):
        #         x = X.dot(w)
        #         probability = self._sigmoid(x)
        #         gradient = np.dot(X.T, (expected_y - probability))
        #         w += self.eta * gradient
    def backpropagation(self, Y_train):
        error = self.gradient_error(Y_train)
        Y_train = np.array([my_dict[value] for value in Y_train])

        for layer in range(len(self.layers) - 1, 0, -1):
            # slope = self.derivative_sigmoid(self.layers[layer].neurons)
            probability = self.sigmoid(self.layers[layer].neurons)
            print(Y_train.shape)
            print(probability.shape)
            print(self.layers[layer-1].neurons.T.shape)
            gradient = np.dot(self.layers[layer - 1].neurons, (Y_train - probability).T)
            self.layers[layer].weights = self.layers[layer].weights + self.eta * gradient
            # d_layer = error * slope
            # print("-------------------------------")
            # print(self.layers[layer].weights)
            # print(self.layers[layer - 1].neurons.T.shape)
            # print(d_layer.shape)
            # print(np.dot(d_layer, self.layers[layer - 1].neurons.T).shape)
            # self.layers[layer].weights = self.layers[layer].weights + np.dot(d_layer, self.layers[layer - 1].neurons.T) * self.eta
            # print("then, ",  self.layers[layer].weights)
            self.layers[layer].bias -= np.sum(d_layer, axis = 1) * self.eta
            error = np.dot(self.layers[layer].weights.T, d_layer)

    def print_shapes(self):
        for layer in range(0, len(self.layers)):
            print("LAYER ", layer)
            print(self.layers[layer].neurons.shape)
            if( layer != 0):
                print(self.layers[layer].weights.shape)
                print(self.layers[layer].bias.shape)

    def fit(self, X_train, Y_train):
        # print(self.layers[1].weights)
        # print(self.layers[1].bias)
        # pass
        print("###############################################")
        print(self.layers[0].neurons)
        for epoch in range(self.epochs):
            # print(epoch)

            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(self.layers[1].neurons)

            for i in range(1, len(self.layers)):
                
                input = np.dot(self.layers[i - 1].neurons.T, self.layers[i].weights.T) + self.layers[i].bias
                if(epoch == self.epochs - 1):
                    print("layer : ", i)
                    print(np.dot(self.layers[i - 1].neurons.T, self.layers[i].weights.T))
                    print(input)
                self.layers[i].neurons = self.sigmoid(input.T)
                # print(self.layers[i].weights)
                # print(self.layers[i].bias)
                if(epoch == self.epochs - 1):
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~é")
                    print(self.layers[i].neurons)
                # print(input.shape)
            # self.print_shapes()
            self.backpropagation(Y_train)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~é")
            # print(self.layers[3].neurons)
            # print(self.layers[3].neurons)
        print("RESULTS")
        print(self.layers[3].neurons)
        # print(Y_train)
        print(self.gradient_error(Y_train))

#output layer is shape of number of labels

class Layer(NeuralNetwork):
    def __init__(self, input, n_neurons, act):
        self.weights = 0
        self.bias = 0
        self.neurons = []
        self.n_neurons = n_neurons
        self.act = act
        for i in range(self.n_neurons):
            self.neurons.append(input[:, i])
        self.neurons = np.array(self.neurons)
    
    def calc_weight():
        self.weights = 0

    def calc_bias():
        self.bias = np.random.rand(1)

def calc_count_mean(data):
    '''
    for each feature, calulate count without blank input
    then calculate average
    '''
    count = [0 for x in range(data.shape[1])]
    mean = [0 for x in range(data.shape[1])]
    print(data.shape[1])
    for column in range(0, data.shape[1]):
        count[column] = np.sum(data[:][column])
        for d in range(data.shape[0]):
            print(data[d][column])
            x = data[d][column]
            if not math.isnan(x):
                mean[column] += x
        mean[column] /= count[column]
    return count, mean

def calc_std(data, count, mean):
    '''
    for each feature, calculate standart deviation, a measure of
    the spread of a distribution
    sdt = sqrt(mean(abs(x - x.mean())**2))
    '''
    std = [0 for x in range(data.shape[1])]
    for column in range(0, data.shape[1]):
        deviations = []
        for d in range(0, data.shape[0]):
            x = data[d][column]
            if not math.isnan(x):
                deviations.append((x - mean[column]) ** 2)
        variance = sum(deviations) / count[column]
        std[column] = math.sqrt(variance)
    return std 

def normalize(count, mean, std, data):
    for column in range(0, data.shape[1]):
        for index in range(0, data.shape[0]):
            # print(data[1][0])
            # print("column = ", column, "index = ", index)
            # print(type(data[index][column]))
            # print(data[index][column])
            # print(type(mean[column]))
            # print(type(std[column]))
            data[index][column] = (data[index][column] - mean[column])/ std[column]

def preprocess_data(data):
    data = data.to_numpy()
    X = data[:, 2:]
    count, mean = calc_count_mean(X)
    std = calc_std(X, count, mean)
    print(X)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~é")
    normalize(count, mean, std, X)
    print(X)
    Y = data[:, 1]
    n = int(X.shape[0] / 10)
    X_train = X[n:, :]
    X_test = X[:n+1, :]
    Y_train = Y[n:]
    Y_test = Y[:n + 1]
    print(X[0][0])
    print(X[1][0])
    print(Y.shape)
    print(Y_test.shape)


    return(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data',
        type=str, help="CSV file containing the dataset",
        default="./dataset/data.csv")
    parser.add_argument('-t', '--train',
        type=bool, help="Specify if you want to train the model",
        default=True)
    parser.add_argument('-p', '--predict',
        type=bool, help="Specify if you want to predict the model",
        default=False)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    X_train, Y_train, X_test, Y_test = preprocess_data(data)

    input_layer = Layer(X_test, X_test.shape[1], "Sigmoid")
    hidden_layer1 = Layer(X_test, 15, "Sigmoid")
    hidden_layer2 = Layer(X_test, 15, "Sigmoid")
    output_layer = Layer(X_test, 1, "Sigmoid")

    NN = NeuralNetwork()
    NN.add(input_layer)
    NN.add(hidden_layer1)
    NN.add(hidden_layer2)
    NN.add(output_layer)
    NN.calculate_weights_bias()
    NN.fit(X_test, Y_test)