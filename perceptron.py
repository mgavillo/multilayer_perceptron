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
        self.eta = 0.9
        self.epochs = 10000
    
    def add(self, layer):
        self.layers.append(layer)
    

    def calculate_weights_bias(self):
        for i in range(1, len(self.layers)):
            self.layers[i].weights = np.random.rand(self.layers[i].n_neurons, self.layers[i - 1].n_neurons)
            self.layers[i].bias = np.random.rand(self.layers[i].n_neurons)
                # self.layers[i].bias = np.random.rand(self.layers[i].n_neurons)

    def relu(self, x):
        return (x > 0) * x
    
    def softmax(self, predicted):
        # print("softmax")
        _max = np.max(predicted)
        predicted -= _max
        # return np.exp(int(predicted)) / np.sum(np.exp(int(predicted)))
        # print("sum = ", np.sum(np.exp(predicted), axis = 0))
        # print("base = ", np.exp(predicted))
        ret = np.exp(predicted) / np.sum(np.exp(predicted), axis=0) 
        # print("ret = ", ret)
        return ret

    def derivative_softmax(self,s):
        return s * (1 - s)

    def derivative_relu(self, x):
        return (x >  0) * 1
    
    def sigmoid(self, X):
        z=np.array(X,dtype=np.float32)
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, X):
        s = self.sigmoid(X)
        return s*(1-s)

    def mean_square_error(self, predicted, expected):
        if predicted.ndim == 1:
            predicted = np.array([predicted])
        return np.sum(np.square(predicted - expected)) / predicted.shape[1]

    def derivative_mean_square_error(self, predicted, expected):
        return predicted - expected


    def print_shapes(self):
        for layer in range(0, len(self.layers)):
            print("LAYER ", layer)
            print(self.layers[layer].neurons.shape)
            if( layer != 0):
                print(self.layers[layer].weights.shape)
                print(self.layers[layer].bias.shape)

    def compute_weight_derivative(self, neuron, Y, X):
        if neuron.neuron_output <= 0:
            result = 0
        else:
        # print(np.shape(y_pred))
            result = 2 * np.mean(np.dot(neuron.neuron_output - Y[:, 0], neuron.inputs.transpose()), axis = 0)
        return result

    def compute_bias_derivative(self, neuron, Y):
        # if y_pred <= 0:
        #     result = 0
        # else:
        result = 2 * np.mean(neuron.neuron_output - Y[:, 0], axis = 0)
        return result

    def fit(self, X_train, Y_train):

        # pass
        Y_train = np.array([my_dict[value] for value in Y_train])
        Y_train = np.expand_dims(Y_train, axis=0)
        # print("neurons = ", self.layers[0].neurons.T)

        print("###############################################")
        self.print_shapes()
        for epoch in range(self.epochs):
            print("EPOCH = ", epoch)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            for i in range(1, len(self.layers)):
                # print("layer = ", i)
                # print(self.layers[i].weights.shape)
                dot_value = np.dot(self.layers[i - 1].neurons.T, self.layers[i].weights.T) + self.layers[i].bias
                # print("before activation = ", dot_value)
                dot_value = dot_value.astype(float)
                if (i != 3):
                    self.layers[i].neurons = self.relu(dot_value).T
                else:
                    self.layers[i].neurons = self.softmax(dot_value).T
                # print("neurons = ", self.layers[i].neurons.T)
                # print("weights = ", self.layers[i].weights.T)
                # print("bias = ", self.layers[i].bias)
            # print("output layer = ", self.layers[-1].neurons)
            # print(self.layers[-1].neurons.shape)
            # print(self.layers[-1].weights.shape)
            # self.backpropagation(Y_train)
            # print("SUM= ", np.sum(self.layers[-1].neurons))
            # print(self.layers[-1].weights)

            error = Y_train[0] - self.layers[-1].neurons[0]
            # error = ((self.layers[-1].neurons[0] - Y_train[0]) / 2) ** 2
            slope = self.derivative_softmax(self.layers[-1].neurons)
            # print("slope = ", slope)
            d_layer = error * slope * self.eta
            self.layers[-1].weights = self.layers[-1].weights + np.dot(d_layer, self.layers[-2].neurons.T)
            self.layers[-1].bias = self.layers[-1].bias + np.sum(d_layer, axis=1)
            # self.layers[-1].weights = self.layers[-1].weights - delta * self.layers[-2].neurons * self.eta
            # print("weights = ", self.layers[-1].weights.shape, "neurons = ", self.layers[-2].neurons.shape, "delta = ", d_layer.shape)
            # print("output layer d, ", d_layer.shape)
            # print(self.layers[-1].bias.shape, "<- bias ")
            #CHANGE BIAS CALCULUS
            for layer in range(len(self.layers) - 2, 0, -1):
                # print("layer = ", layer)
                slope = self.derivative_relu(self.layers[layer].neurons)
                error = np.dot(self.layers[layer + 1].weights.T, d_layer)
                d_layer = slope * error * self.eta
                # print("d layer =", d_layer.shape)
                # print("weights = ", self.layers[layer].weights.shape, "neurons = ", self.layers[layer- 1].neurons.shape, "delta = ", d_layer.shape)

                self.layers[layer].weights = self.layers[layer].weights + np.dot(d_layer, self.layers[layer - 1].neurons.T)
                # print("BIAS = ", self.layers[layer].bias)
             
                self.layers[layer].bias = self.layers[layer].bias + np.sum(d_layer, axis=1)
            # print(self.layers[-1].weights)
        print("RESULTS")
        print(self.layers[-1].neurons)
        print(Y_train)
        print(Y_train - self.layers[-1].neurons)

#output layer is shape of number of labels

class Layer(NeuralNetwork):
    def __init__(self, input, n_neurons, act):
        self.weights = []
        self.bias = []
        self.neurons = []
        self.n_neurons = n_neurons
        self.act = act
        for i in range(self.n_neurons):
            self.neurons.append(input[:, i])
        self.neurons = np.array(self.neurons)
        print(self.neurons.shape)
    
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
    # print(data.shape[1])
    for column in range(0, data.shape[1]):
        count[column] = np.sum(data[:][column])
        for d in range(data.shape[0]):
            # print(data[d][column])
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
    data = data.T
    # count, mean = calc_count_mean(data)
    # std = calc_std(data, count, mean)
    # print(data.shape[0])
    for column in range(0, data.shape[0]):
        # print("column = ", data[column])
        
        _max = np.max(data[column])
        _min = np.min(data[column])
        # print("MAX=", _max)
        # print("min = ", _min)
        for i, x  in enumerate(data[column]):
        # for i in range(0, data.shape[0]):
            # print(data[1][0])
            # print("column = ", column, "index = ", index)
            # print(type(data[index][column]))
            # print(data[index][column])
            # print(type(mean[column]))
            # print(type(std[column]))
            data[column][i] = (x - _min) / (_max - _min)
            # data[i][column] = (data[i][column] - mean[column])/ std[column]
    data = data.T
    return(data)

def preprocess_data(data):
    data = data.to_numpy()
    X = data[:, 2:]
    count, mean = calc_count_mean(X)
    std = calc_std(X, count, mean)
    # print(X)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~é")
    X = normalize(count, mean, std, X)
    # print(X)
    Y = data[:, 1]
    n = X.shape[0] - int(X.shape[0] / 4)
    X_train = X[n:, :]
    X_test = X[:n+1, :]
    Y_train = Y[n:]
    Y_test = Y[:n + 1]
    # print(X[0][0])
    # print(X[1][0])
    # print(Y.shape)
    # print(Y_test.shape)


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
    print("WSH ALORS")

    input_layer = Layer(X_test, X_test.shape[1], "Sigmoid")
    hidden_layer1 = Layer(X_test, 18, "Sigmoid")
    hidden_layer2 = Layer(X_test, 15, "Sigmoid")
    output_layer = Layer(X_test, 1, "Sigmoid")

    NN = NeuralNetwork()
    NN.add(input_layer)
    NN.add(hidden_layer1)
    NN.add(hidden_layer2)
    NN.add(output_layer)
    NN.calculate_weights_bias()
    NN.fit(X_test, Y_test)