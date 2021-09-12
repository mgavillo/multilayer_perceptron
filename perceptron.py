import numpy as np
import pandas as pd
import argparse
import math
import random
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
        self.eta = 0.01
        self.epochs = 1
        self.mini_batch_size = 4
      
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
        print("softmax")
        # print(predicted)
        print(predicted[:, 0].shape)

        _max = np.max(predicted[:, 0])
        predicted[:, 0] = predicted[:, 0] - _max
        predicted[:, 0] = predicted[:, 0].astype(float)

        _max = np.max(predicted[:, 1])
        predicted[:, 1] = predicted[:, 1] - _max
        predicted[:, 1] = predicted[:, 1].astype(float)
        print(predicted[:, 0])
        print(predicted[:, 1])
        print(predicted[:, 2])
        print(predicted[:, 3])
        print(predicted[0, :])
        print("wsh")
        print(predicted)
        ret = [np.exp(predicted[:, i]) / np.sum(np.exp(predicted[:, i])) for i in range(predicted.shape[1])]
        print("ret = ", ret)
        return np.array(ret)

    def derivative_softmax(self,S):
        # s[range(y.shape[0]), y] -= 1
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
                print(self.layers[layer].dot_value.shape)

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

        print("init")
        for i in range(self.layers[0].n_neurons):
            self.layers[0].neurons.append(X_train[:, i])
        self.layers[0].neurons = np.array(self.layers[0].neurons)
        # print(self.layers[0].neurons.shape)
        # print(X_train.shape[0])
        for i in range(1, 4):
            self.layers[i].neurons = np.zeros([self.layers[i].n_neurons, X_train.shape[0]])
            print(self.layers[i-1].neurons.shape, self.layers[i].weights.shape )
        # print(self.layers[3].neurons.shape)
# pass
        # self.print_shapes()
        Y_train = np.array([my_dict[value] for value in Y_train])
        Y_train = np.expand_dims(Y_train, axis=0)
        # print("neurons = ", self.layers[0].neurons.T)
        n = self.layers[0].neurons.shape[1]
        mini_batch_size = 4
        # print(self.layers[0].neurons)
        # print("iiiiiiiiiiiiiiiii")
        # print(self.layers[0].neurons[1])
        # print("ooooooooooooooo")
        # print(self.layers[0].neurons[:][1])
        # print("N ==== ", n)
        # print("###############################################")
        # self.print_shapes()
        X_train = X_train.T
        X = X_train
        YY = Y_train
        for epoch in range(self.epochs):
            print("EPOCH = ", epoch)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            x = np.arange(n)
            np.random.shuffle(X_train)
            X_train = X_train[:, x]
            Y_train = Y_train[:, x]
            # random.shuffle(self.layers[0].neurons)
            mini_batches = [
                X_train[:, k:k+mini_batch_size]
                for k in np.arange(0, n, mini_batch_size)]
            Y_trains = [
                Y_train[:, k:k+mini_batch_size]
                for k in np.arange(0, n, mini_batch_size)]
            for mini_batch, Y in zip(mini_batches, Y_trains):
                self.layers[0].neurons = mini_batch
                if mini_batch.shape[1] == 3:
                    break
                for i, layer in enumerate(self.layers):
                    if (i != 0):
                        self.layers[i].nabla_b = [np.zeros(b.shape) for b in self.layers[i].bias]                #Initialize bias matrix with 0's
                        self.layers[i].nabla_w = [np.zeros(w.shape) for w in self.layers[i].weights]               #Initialize weights matrix with 0's
                        layer.dot_value = np.zeros([mini_batch_size,layer.weights.shape[0]])
                        print("i ========",i)
                        # print(layer.bias)
                        layer.dot_value = np.dot(layer.weights, self.layers[i - 1].neurons) + np.dot(np.array([layer.bias]).T, np.ones([1, mini_batch_size]))
                        layer.dot_value = layer.dot_value.astype(float)
                        if (i != 3):
                            layer.neurons = self.relu(layer.dot_value)
                        else:
                            print(layer.dot_value.shape)
                            layer.neurons = self.softmax(layer.dot_value)
                            print(layer.neurons.shape)
                        print(layer.neurons)
                Y_output = []
                # print(Y.shape)
                # print(Y[0])
                for z in range(Y.shape[1]):
                    print(z)
                    if Y[0][z] == 0:
                        b = 1
                    else:
                        b = 0
                    Y_output.append([Y[0][z], b])
                Y_output = np.array(Y_output).T
                error = Y_output - self.layers[-1].neurons

                print("feed forward made")
                # self.print_shapes()
                # error = np.array([error])
                # slope = self.derivative_softmax(self.layers[-1].neurons)
                # slope = 1
                # print("slope")
                # print(slope)
                d_layer = error * self.eta
                # print("d_layer output", d_layer)
                # print("weights before")
                # print(self.layers[-1].weights)
                # print(d_layer.shape)
                # print("+")
                # print(np.dot(d_layer, self.layers[-2].neurons.T))
                # self.layers[-1].weights = [w - nw for w, nw in zip(self.layers[-1].weights, self.layers[-1].nabla_w)]
                self.layers[-1].weights = self.layers[-1].weights + np.dot(d_layer, self.layers[-2].neurons.T)
                # self.print_shapes()

                self.layers[-1].bias = self.layers[-1].bias + np.sum(d_layer, axis=1)
                # self.layers[-1].bias = [b - nb for b, nb in zip(self.layers[-1].bias, self.layers[-1].nabla_b)]

                # print("after weights")
                # print(self.layers[-1].weights)
                # print(layer.shape)
                dot_value = np.dot(self.layers[-1].weights, self.layers[-2].neurons) + np.dot(np.array([self.layers[-1].bias]).T, np.ones([1, mini_batch_size]))
                output = self.softmax(dot_value)
                # print(dot_value)
                # print("output")
                print(output)
                print("expected")
                print(Y_output)
                print("~~~~~~~~~~~~~~~~~~~~~~")
                

                for layer in range(len(self.layers) - 2, 0, -1):
                    slope = self.derivative_relu(self.layers[layer].dot_value).T
                    # print(self.layers[layer + 1].weights)
                    error = np.dot(self.layers[layer + 1].weights.T, d_layer)
                    d_layer = error * self.eta
                    self.layers[layer].weights = self.layers[layer].weights + np.dot(d_layer, self.layers[layer - 1].neurons.T)
                    self.layers[layer].bias = self.layers[layer].bias + np.sum(d_layer, axis=1)

        print("RESULTS")
        self.layers[0].neurons = X
        # print(self.layers[0].neurons)
        for i in range(1, len(self.layers)):
            dot_value = np.dot(self.layers[i].weights, self.layers[i - 1].neurons) + np.dot(np.array([self.layers[i].bias]).T, np.ones([1, X.shape[1]]))
            dot_value = dot_value.astype(float)
            if (i != 3):
                self.layers[i].neurons = self.relu(dot_value)
            else:
                self.layers[i].neurons = self.softmax(dot_value)
        output = []
        print(self.layers[-1].neurons.shape)
        for i in range(self.layers[-1].neurons.shape[1]):
            if(self.layers[-1].neurons[0][i] < self.layers[-1].neurons[1][i]):
                output.append(0)
            elif(self.layers[-1].neurons[0][i] == self.layers[-1].neurons[1][i]):
                output.append(2)
            else:
                output.append(1)
        print(output)
        print(YY)
        self.print_shapes()

        

#output layer is shape of number of labels

class Layer(NeuralNetwork):
    def __init__(self, input, n_neurons, act):
        self.weights = []
        self.bias = []
        self.neurons = []
        self.n_neurons = n_neurons
        self.act = act
        self.dot_value = []

        # print(self.neurons.shape)

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
    output_layer = Layer(X_test, 2, "Sigmoid")

    NN = NeuralNetwork()
    NN.add(input_layer)
    NN.add(hidden_layer1)
    NN.add(hidden_layer2)
    NN.add(output_layer)
    NN.calculate_weights_bias()
    NN.fit(X_test, Y_test)