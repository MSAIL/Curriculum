import numpy as np
from scipy.special import expit
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='/Users/nanddalal/tmp/')

x = mnist.data
y = mnist.target
xy = np.c_[x,y]
np.random.shuffle(xy)

num_train = 10000
num_test = 2000
num_hidden_units = 100

train_x = xy[0:num_train, 0:784]
train_y = xy[0:num_train, -1]
test_x = xy[num_train:num_train+num_test, 0:784]
test_y = xy[num_train:num_train+num_test, -1]

def train_and_test(model, xtrain, ytrain, xtest, ytest):
    model = model.fit(xtrain, ytrain, xtest, ytest)
    output = model.predict(xtest)
    print (1.0 * sum([i==j for (i, j) in zip(output, ytest)])) / len(ytest)

train_y_probs = np.zeros((num_train, 10))
for i in range(num_train):
    train_y_probs[i][int(train_y[i])] = 1.0

class nnet:
    def __init__(self):
        self.w1 = np.random.randn(784, num_hidden_units)
        self.b1 = np.random.randn(num_hidden_units)
        self.w2 = np.random.randn(num_hidden_units, 10)
        self.b2 = np.random.randn(10)
    
    @staticmethod
    def sigmoid(x):
        return expit(x)

    @staticmethod
    def dsigmoid(x):
        return expit(x) * (1 - expit(x))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    def forward_propagate(self, x):
        self.a1 = x
        self.z2 = np.dot(self.a1, self.w1) + self.b1
        self.a2 = nnet.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2) + self.b2
        self.a3 = nnet.softmax(self.z3)

    def backward_propagate(self, y):
        self.lr = 0.001

        del3 = self.a3 - y
        w2grad = np.dot(self.a2.reshape(num_hidden_units, 1), del3.reshape(1, 10))
        b2grad = w2grad.sum(axis=0)
        self.w2 -= self.lr*w2grad
        self.b2 -= self.lr*b2grad

        del2 = np.dot(del3, self.w2.T) * nnet.dsigmoid(self.z2)
        w1grad = np.dot(self.a1.reshape(784, 1), del2.reshape(1, num_hidden_units))
        b1grad = w1grad.sum(axis=0)
        self.w1 -= self.lr*w1grad
        self.b1 -= self.lr*b1grad

    def fit(self, train_x, train_y, test_x, test_y):
        for it in range(1000):
            if it % 10 == 0:
                print "Epoch", it
                output = self.predict(test_x)
                print (1.0 * sum([i==j for (i, j) in zip(output, test_y)])) / len(test_y)

            for i in range(num_train):
                self.forward_propagate(train_x[i])
                self.backward_propagate(train_y[i])
        
        return self

    def predict(self, test_x):
        preds = np.zeros(num_test)
        for i in range(num_test):
            self.forward_propagate(test_x[i])
            preds[i] = np.argmax(self.a3)
        return preds
    
    def visualize(self):
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.axis('off')
            plt.set_cmap('gray')
            plt.imshow(model.weights[:,i].reshape(28, 28))

model = nnet()
train_and_test(model, train_x, train_y_probs, test_x, test_y)
