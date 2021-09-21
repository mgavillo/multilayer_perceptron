import numpy as np
import pandas as pd
import argparse
import math
import random
from tqdm import tqdm
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
        self.epochs = 10000
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
            # print(predicted[:, i])
            # print(f)
            if np.sum(f) == 0:
                exit 
            ret.append(f / np.sum(f))
        # print(ret)
        return np.array(ret).T

    def derivative_softmax(self,S):
        # s[range(y.shape[0]), y] -= 1
        # print("SSSSSSSSSSSSSSSSSS = " , S[0])
        # S[1] = S[1] * (1 - S[1])
        # S[0] = -S[0] * S[1] 
        return S * (1 - S)


    
    def sigmoid(self, X):
        z=np.array(X,dtype=np.float32)
        # print(z)
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, X):
        s = self.sigmoid(X)
        return s*(1-s)

    def print_shapes(self):
        for layer in range(0, len(self.layers)):
            print("LAYER ", layer)
            print(self.layers[layer].neurons.shape)
            if( layer != 0):
                print(self.layers[layer].weights.shape)
                print(self.layers[layer].bias.shape)
                # print(self.layers[layer].dot_value.shape)

    def feed_forward(self, layers):
        # print(self.layers[0].neurons)
        for i, layer in enumerate(layers):
            if (i != 0):
                # print("feed forward", i)
                # print(layer.weights.shape)
                # print(self.layers[i -1].neurons.shape)
                layer.dot_value = np.dot(layer.weights, self.layers[i - 1].neurons) 
                # print("before before", layer.dot_value)
                layer.dot_value += np.dot(np.array([layer.bias]).T, np.ones([1, self.mini_batch_size]))
                # print("before", layer.dot_value)

                # print("after ",layer.dot_value)
                # print("bias =", np.dot(np.array([layer.bias]).T, np.ones([1, self.mini_batch_size])))
                layer.dot_value = layer.dot_value.astype(float)
                if(i == 1):
                    layer.neurons = self.sigmoid(layer.dot_value)
                elif (i == 2):
                    layer.neurons = self.relu(layer.dot_value)
                else:
                    layer.neurons = self.softmax(layer.dot_value)
                # print("after = ", layer.neurons)

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
        # print("BACKPROP\n")
        Y_output = self.y_to_dual(Y)
        # print(Y_output, "wsh",  self.layers[-1].neurons)
        error =  Y_output - self.layers[-1].neurons
        # print("error =", error)
        slope = self.derivative_softmax(self.layers[-1].neurons)
        d_layer = np.array(error * slope, dtype=np.float64)
        # d_layer *= 10** 7 
        # print("neurons", self.layers[-1].neurons)
        # print("Y_output = ", Y_output)
        # print("d_layer", d_layer)
        # print("error =", np.mean(d_layer, axis = 1))
        # print(self.layers[-1].weights)
        # print("bias =", self.layers[-1].bias)
        # print(self.layers[-1].weights.dtype)
        # print("dlayer mean = ", np.mean(d_layer, axis=1), np.mean(d_layer, axis=1).shape)
        # print(np.dot(d_layer, self.layers[-2].neurons.T).shape)
        self.layers[-1].weights += self.eta * np.dot(d_layer, self.layers[-2].neurons.T)
   
        self.layers[-1].bias += self.eta * np.mean(d_layer, axis = 1)
        # print(self.layers[-1].bias)
        # print("mean")
        # print(np.mean(d_layer, axis=1))
        # print(self.layers[-1].weights)
        # print("bias =", self.layers[-1].bias)
        for layer in range(len(self.layers) - 2, 0, -1):
            if(layer == 1):
                slope = self.derivative_sigmoid(self.layers[layer].dot_value)
            else:
                slope = self.derivative_relu(self.layers[layer].dot_value)
            error = np.dot(self.layers[layer + 1].weights.T, d_layer)
            d_layer = np.array(error * slope, dtype=np.float64)
            self.layers[layer - 1].neurons = np.array(self.layers[layer - 1].neurons, dtype=np.float64)
            self.layers[layer].weights += self.eta * np.dot(d_layer, self.layers[layer - 1].neurons.T)
            self.layers[layer].bias += self.eta * np.mean(d_layer, axis = 1)
    
    # def gradient_check(self):

    def test_back_prop(self):
        self.feed_forward(self.layers)
        print(self.layers[-1].neurons)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def get_results(self, X, Y):
        print("RESULTS")
        self.layers[0].neurons = X.T
        Y = np.array([my_dict[value] for value in Y])

        Y = self.y_to_dual(Y)
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
    
    def fit(self, X_train, Y_train):

        print("init")
        for i in range(self.layers[0].n_neurons):
            self.layers[0].neurons.append(X_train[:, i])
        self.layers[0].neurons = np.array(self.layers[0].neurons)
        for i in range(1, len(self.layers)):
            print(i)
            self.layers[i].neurons = np.zeros([self.layers[i].n_neurons, X_train.shape[0]])
        self.print_shapes()

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
                # see with end of batches*
                
                if mini_batch.shape[1] != self.mini_batch_size:
                    break
                self.feed_forward(self.layers)
                self.back_prop(y_batch)
                # self.test_back_prop()
        # self.get_results(X, Y)

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

    # print(X_test.shape[1])
    # X_test = np.array([[1, 0, 0, 1], [0, 1, 0, 1]]).T
    # print(X_test.shape[1])
    # Y_test = np.array([1, 1, 0, 0])
    input_layer = Layer(X_test, X_test.shape[1], "Sigmoid")
    hidden_layer1 = Layer(X_test, 16, "Sigmoid")
    hidden_layer2 = Layer(X_test, 7, "Sigmoid")
    output_layer = Layer(X_test, 2, "Sigmoid")

    NN = NeuralNetwork()
    NN.add(input_layer)
    NN.add(hidden_layer1)
    NN.add(hidden_layer2)
    NN.add(output_layer)
    NN.calculate_weights_bias()
    NN.fit(X_test, Y_test)
    NN.get_results(X_test, Y_test)