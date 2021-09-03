import numpy as np
import pandas as pd
import argparse
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
        self.eta = 1 #set up the proper Learning Rate!!
        self.epochs = 1000
    
    def add(self, layer):
        self.layers.append(layer)
    

    def calculate_weights_bias(self):
        for i in range(0, len(self.layers)):
            if i != 0:
                self.layers[i].weights = np.random.rand(self.layers[i].n_neurons, self.layers[i - 1].n_neurons)
                self.layers[i].bias = np.random.rand(self.layers[i].n_neurons)

    def sigmoid(self, X):
        z=np.array(X,dtype=np.float32)
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, X):
        s = self.sigmoid(X)
        return s*(1-s)

    def gradient_error(self, Y_train):
        Y_train = np.array([my_dict[value] for value in Y_train])
        print("\n")
        print(Y_train)

        print(Y_train.shape)
        return Y_train - self.layers[3].neurons

    # def update_weights(self):
    #     for i in range(0, len(self.layers)):
    #         if i != 0:
    #             self.layers[i].weights = np.dot(sigmoid(self.layers[i - 1].neurons).T, )

    def fit(self, X_train, Y_train):
        for epoch in range(self.epochs):
            # print(epoch)
            for i in range(0, len(self.layers)):
                if i != 0:
                    print("neurons = ", self.layers[i - 1].neurons.T.shape)
                    print("weights = ", self.layers[i].weights.T.shape)
                    input = np.dot(self.layers[i - 1].neurons.T, self.layers[i].weights.T) + self.layers[i].bias
                    self.layers[i].neurons = self.sigmoid(input.T)
                    print(input.shape)
            error = self.gradient_error(Y_train)

            slope_output_layer = self.derivative_sigmoid(self.layers[3].neurons).T
            slope_hidden_layer1 = self.derivative_sigmoid(self.layers[1].neurons).T
            slope_hidden_layer2 = self.derivative_sigmoid(self.layers[2].neurons).T

            d_output = error * slope_output_layer
            print("cucul")
            print(d_output.shape)
            print(self.layers[3].weights.shape)
            Error_at_hidden_layer2 = np.dot(d_output, self.layers[3].weights.T)
            
            print(Error_at_hidden_layer2.shape)
            print(slope_hidden_layer2.shape)
            d_hiddenlayer2 =  Error_at_hidden_layer2 * slope_hidden_layer2
            
            Error_at_hidden_layer1 = np.dot(d_hiddenlayer2.T , self.layers[2].weights)
            d_hiddenlayer1 = Error_at_hidden_layer1 * slope_hidden_layer1

            self.layers[3].weigths = self.layers[3].weights + np.dot(self.layers[3].neurons, d_output) * self.eta
            self.layers[2].weights = self.layers[2].weights + np.dot(self.layers[2].neurons, d_hiddenlayer2) * self.eta
            self.layers[1].weights = self.layers[1].weights + np.dot(self.layers[1].enurons, d_hiddenlayer1) * self.eta
            
            self.layers[3].bias += sum(d_ouptut, axis=0) * self.eta
            self.layers[2].bias += sum(d_hiddenlayer2, axis=0) * self.eta
            self.layers[1].bias += sum(d_hiddenlayer1, axis=0) * self.eta
        


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

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)

def preprocess_data(data):
    data = data.to_numpy()
    X = data[:, 2:]
    Y = data[:, 1]
    n = int(X.shape[0] / 10)
    X_train = X[n:, :]
    X_test = X[:n+1, :]
    Y_train = Y[n:]
    Y_test = Y[:n + 1]
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