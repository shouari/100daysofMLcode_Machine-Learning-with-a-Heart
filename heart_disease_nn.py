'''The code was built thanks to the ispiration of :
        Daphne Cornelisse : https://www.kaggle.com/daphnecor/week-1-3-layer-nn?scriptVersionId=2461384
        Venelin Valkov : https://medium.com/@curiousily/tensorflow-for-hackers-part-iv-neural-network-from-scratch-1a4f504dfa8
        Florian Courtial  : https://matrices.io/deep-neural-network-from-scratch/

         '''
import matplotlib.pyplot as plt
import numpy as np



class NeuralNetwork:
    def __init__(self, X, y, training_set, nodes_layer_1, nodes_layer_2, n_class=1, regul_fact=0.9,
                 epoch=500, learning_rate=0.001):
        self.n_features = np.size(X, axis=1)  # number of features == number of input nodes
        self.n_class = n_class  # Number of classes == number of output nodes
        self.m_train = int(float(X.shape[0]) * training_set)
        self.nodes_layer_1 = nodes_layer_1  # number of nodes in hidden layer 1
        self.nodes_layer_2 = nodes_layer_2  # number of nodes in hidden layer 1

        # Initializing the weights and biases for each layer
        self.model = self.init_weights()

        self.regul_fact = regul_fact   # Regularization factor to address overfitting
        self.epoch = epoch
        self.learning_rate = learning_rate

    # Defining the activation functions and derivatives
    def tanh_prime(self, x):
        return 1.0 - np.tanh(x) ** 2

    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_prime(self,x):
        return x * (1.0 - x)

    def relu(self,x):
        return x * (x > 0)

    def relu_prime(self,x):
        return 1. * (x > 0)


    #  function to initialize the weights and biases
    def init_weights(self):
        np.random.seed(1)           # We seed the weights as to get the same initial weights &
        # biases each time we run the training

        # To initialize the weights, we use the He-et-al Initialization method with reference to
        # the article of Aditya Ananthram (
        # https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e)

        w1 = np.random.randn(self.n_features, self.nodes_layer_1) * np.sqrt(2 /
                                                                                 self.n_features)
        b1 = np.random.randn(1, self.nodes_layer_1) * np.sqrt(2 / self.n_features)

        w2 = np.random.randn(self.nodes_layer_1, self.nodes_layer_2) * np.sqrt(2/self.nodes_layer_1)
        b2 = np.random.randn(1, self.nodes_layer_2) * np.sqrt(2/self.nodes_layer_1)

        w3 = np.random.randn(self.nodes_layer_2, self.n_class) * np.sqrt(2/self.nodes_layer_2)
        b3 = np.random.randn(1, self.n_class) * np.sqrt(2/self.nodes_layer_2)

        model = {'W1': w1, 'B1': b1, 'W2': w2, 'B2': b2, 'W3': w3, 'B3': b3}
        return model

    def forward_prop(self, X, model):  # forward propagation algorithm

        # Load model
        w1, b1 = model['W1'], model['B1']
        w2, b2 = model['W2'], model['B2']
        w3, b3 = model['W3'], model['B3']

        # First layer
        z1 = np.dot(X, w1) + b1
        h1 = self.relu(z1)  # activation function for the first layer
        # we add the 1 unit (bias) to the output layer

        # Second layer

        z2 = np.dot(h1, w2) + b2
        h2 = self.relu(z2)
        # we add the 1 unit (bias) to the output layer

        # output layer we use softmax function (imported from sklearn library)
        z3 = np.dot(h2, w3) + b3
        h3 = self.sigmoid(z3)

        # Storing all values (will note a0 as input features in our cache list)
        cache = {'a0': X, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3, 'h3': h3}
        return cache

    def loss(self, y, y_predict, model):  # cost function computation with regularization factor
        # lambda
        m = y.shape[0]
        w1, b1 = model['W1'], model['B1']
        w2, b2 = model['W2'], model['B2']
        w3, b3 = model['W3'], model['B3']
        J = (-1/m) * np.sum(np.multiply(y, np.log(y_predict)) + np.multiply((1 - y),
                                                                                 np.log(1 - y_predict)))

        return J

    def gradients(self, y, cache, model):  # this function computes the gradients
        # Loading weights from model
        w1, b1 = model['W1'], model['B1']
        w2, b2 = model['W2'], model['B2']
        w3, b3 = model['W3'], model['B3']

        # Loading results of forward propagation
        X, z1, h1, z2, h2, z3, h3 = cache['a0'], cache['z1'], cache['h1'], cache['z2'],  \
                                    cache['h2'],cache['z3'], cache['h3']

        # Loss derivative with respect to output layer:
        dz3 = np.multiply((h3 - y), self.sigmoid_prime(h3))

        # Loss derivative with respect to weights of output layer
        dw3 = (1 / self.m_train) * np.dot(h2.T, dz3) + (self.regul_fact / self.m_train) * w3
        # Loss derivative with respect of biases of output layer
        db3 = (1 / self.m_train) * np.sum(dz3, axis=0) + (self.regul_fact / self.m_train) * b3

        # Loss derivative with respect to Hidden layer 2:
        dz2 = np.multiply(np.dot(dz3, w3.T), self.relu_prime(h2))
        # Loss derivative with respect to weights of Hidden layer 2
        dw2 = (1 / self.m_train) * np.dot(h1.T, dz2) + (self.regul_fact / self.m_train) * w2
        # Loss derivative with respect of biases of Hidden layer 2
        db2 = (1 / self.m_train) * np.sum(dz2, axis=0) + (self.regul_fact / self.m_train) * b2

        # Loss derivative with respect to Hidden layer 1:
        dz1 = np.multiply(np.dot(dz2, w2.T), self.relu_prime(h1))
        # Loss derivative with respect to weights of Hidden layer 1
        dw1 = (1 / self.m_train) * np.dot(X.T, dz1) + (self.regul_fact / self.m_train) * w1
        # Loss derivative with respect of biases of Hidden layer 1
        db1 = (1 / self.m_train) * np.sum(dz1, axis=0) + (self.regul_fact / self.m_train) * b1

        # staving gradients:
        grads = {'dW3': dw3, 'db3': db3, 'dW2': dw2, 'db2': db2, 'dW1': dw1, 'db1': db1}
        return grads

    def backward_prop(self, model, grads):
        w1, b1 = model['W1'], model['B1']
        w2, b2 = model['W2'], model['B2']
        w3, b3 = model['W3'], model['B3']

        # Updating the weights while back propagating
        w1 -= self.learning_rate * grads['dW1']
        b1 -= self.learning_rate * grads['db1']
        w2 -= self.learning_rate * grads['dW2']
        b2 -= self.learning_rate * grads['db2']
        w3 -= self.learning_rate * grads['dW3']
        b3 -= self.learning_rate * grads['db3']

        # saving parameters
        model = {'W1': w1, 'B1': b1, 'W2': w2, 'B2': b2, 'W3': w3, 'B3': b3}
        return model

    def _predict(self, X):

        c = self.forward_prop(X, self.model)
        prediction = c['h3']

        return prediction

    def _accuracy(self, X, y):

        pred = self._predict(X)
        error = np.sum(np.abs(pred - y))
        return (X.shape[0] - error) / X.shape[0] * 100

    def _train_model(self, X, y):

        losses = []
        accuracies = []
        model = self.model
        for i in range(0, self.epoch):
            # propagate forward:
            cache = self.forward_prop(X, model)
            # back prop
            grads = self.gradients(y, cache, model)
            model_new = self.backward_prop(model, grads)
            self.model = model_new
            if i % 100 == 0:
                h3 = cache['h3']
                print(' The loss after iteration: ', i, 'is ', self.loss(y, h3, model))
                pred = self._predict(X)
                print("Accuracy after iteration: ", i, 'is  %3.2f' %self._accuracy(X, y),
                      '%')

                losses.append(self.loss(y, pred, model))
                accuracies.append(self._accuracy(X, y))
         # Plotting the loss and accuracy after training to monitor
        plt.subplot(2, 1, 1)
        plt.plot(losses)
        plt.title('Performance of model during the training')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(accuracies)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.show()
        
        self.model = model
        return model, losses, accuracies

    def _test_model(self, X, y):

        pred = self._predict(X)
        cost = self.loss(y, pred, self.model)
        score = self._accuracy(X, y)
        
        # printing loss and accuracy on the testing data
        print('\n' + 'Loss on the test set is %s' % cost)
        print('\n' + ' Accuracy for the testing set is  %3.2f  ' % score, '%')
        print(pred[:,0])

        return pred, cost, score

