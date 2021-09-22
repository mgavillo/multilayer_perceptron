import numpy as np
import pandas as pd
import argparse
import math
import random
from tqdm import tqdm
my_dict = {'M': 1, "B": 0}

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.b = []
        self.phi = []
        self.mu = []
        self.eta = 0.01
        self.epochs = 20000
        self.mini_batch_size = 40
      
    def add(self, layer):
        self.layers.append(layer)
    

    def calculate_weights_bias(self):
        for i in range(1, len(self.layers)):
            self.layers[i].weights = np.random.rand(self.layers[i].n_neurons, self.layers[i - 1].n_neurons) / 10
            self.layers[i].bias = np.random.rand(self.layers[i].n_neurons) / 10

    def relu(self, x):
        return (x > 0) * x

    def derivative_relu(self, x):
        return (x >  0) * 1
  
    def softmax(self, predicted):
        ret = []
        for i in range(predicted.shape[1]):
            f = np.exp(predicted[:, i])
            if np.sum(f) == 0:
                exit 
            ret.append(f / np.sum(f))
        return np.array(ret).T

    def derivative_softmax(self,S):
        return S * (1 - S)
    
    def sigmoid(self, X):
        z=np.array(X,dtype=np.float32)
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, X):
        s = self.sigmoid(X)
        return s*(1-s)

    def feed_forward(self, layers):
        for i, layer in enumerate(layers):
            if (i != 0):
                layer.dot_value = np.dot(layer.weights, self.layers[i - 1].neurons) 
                layer.dot_value += np.dot(np.array([layer.bias]).T, np.ones([1, self.mini_batch_size]))
                layer.dot_value = layer.dot_value.astype(float)
                layer.neurons = layer.act(layer.dot_value)

    def y_to_dual(self, Y):
        Y_output = []
        for z in range(len(Y)):
            if Y[z] == 0:
                b = 1
            else:
                b = 0
            Y_output.append([Y[z], b])
        return np.array(Y_output).T

    def back_prop(self, Y):
        Y_output = self.y_to_dual(Y)
        error =  Y_output - self.layers[-1].neurons
        slope = self.layers[-1].dact(self.layers[-1].neurons)
        d_layer = np.array(error * slope, dtype=np.float64)

        self.layers[-1].weights += self.eta * np.dot(d_layer, self.layers[-2].neurons.T)
        self.layers[-1].bias += self.eta * np.mean(d_layer, axis = 1)
        for layer in range(len(self.layers) - 2, 0, -1):
            slope = self.layers[layer].dact(self.layers[layer].dot_value) 
            error = np.dot(self.layers[layer + 1].weights.T, d_layer)
            d_layer = np.array(error * slope, dtype=np.float64)
            self.layers[layer - 1].neurons = np.array(self.layers[layer - 1].neurons, dtype=np.float64)
            self.layers[layer].weights += self.eta * np.dot(d_layer, self.layers[layer - 1].neurons.T)
            self.layers[layer].bias += self.eta * np.mean(d_layer, axis = 1)
    
    def get_results(self, X, Y):
        print("RESULTS")
        self.layers[0].neurons = X.T
        Y = np.array([my_dict[value] for value in Y])

        # Y = self.y_to_dual(Y)
        self.mini_batch_size = X.shape[0]
        self.feed_forward(self.layers)
        output = []
        for i in range(self.layers[-1].neurons.shape[1]):
            if(self.layers[-1].neurons[0][i] < self.layers[-1].neurons[1][i]):
                output.append(0)
            elif(self.layers[-1].neurons[0][i] == self.layers[-1].neurons[1][i]):
                output.append(2)
            else:
                output.append(1)
        print(output)
        print("\n\n.")
        print(Y)
        print(".")
        print(Y- output)
        print(np.mean(Y- output)/ X.shape[0])
    
    def fit(self, X_train, Y_train):

        for i in range(self.layers[0].n_neurons):
            self.layers[0].neurons.append(X_train[:, i])
        self.layers[0].neurons = np.array(self.layers[0].neurons)
        for i in range(1, len(self.layers)):
            self.layers[i].neurons = np.zeros([self.layers[i].n_neurons, X_train.shape[0]])

        Y_train = np.array([my_dict[value] for value in Y_train])
        X_train = X_train.T
        X = X_train
        Y = Y_train
        for _ in tqdm(range(self.epochs), desc="Epochs"):
            x = np.arange(X_train.shape[1])
            np.random.shuffle(x)
            X_train = X_train[:, x]
            Y_train = Y_train[x]
            mini_batches = [
                X_train[:, k:k+self.mini_batch_size]
                for k in np.arange(0, X_train.shape[1], self.mini_batch_size)]
            Y_trains = [
                Y_train[k:k+self.mini_batch_size]
                for k in np.arange(0, X_train.shape[1], self.mini_batch_size)]
            
            for mini_batch, y_batch in zip(mini_batches, Y_trains):
                self.layers[0].neurons = mini_batch
                
                if mini_batch.shape[1] != self.mini_batch_size:
                    self.mini_batch_size = mini_batch.shape[1]
                    # break
                self.feed_forward(self.layers)
                self.back_prop(y_batch)
        # self.get_results(X, Y)

class Layer(NeuralNetwork):
    def __init__(self, input, n_neurons, act=""):
        self.weights = []
        self.bias = []
        self.neurons = []
        self.n_neurons = n_neurons
        if(act != ""):
            self.act = getattr(self, act)
            self.dact = getattr(self, "derivative_" + act)
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


    input_layer = Layer(X_test, X_test.shape[1])
    hidden_layer1 = Layer(X_test, 16, "sigmoid")
    hidden_layer2 = Layer(X_test, 7, "relu")
    output_layer = Layer(X_test, 2, "softmax")

    NN = NeuralNetwork()
    NN.add(input_layer)
    NN.add(hidden_layer1)
    NN.add(hidden_layer2)
    NN.add(output_layer)
    NN.calculate_weights_bias()
    NN.fit(X_test, Y_test)
    NN.get_results(X_test, Y_test)