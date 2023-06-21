# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the 
# Logistic Regression algorithm. Insert your code into the various functions 
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. 


import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# import pickle 


class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the model weights
    """
    def __init__(self):
        self.w = []
        pass
        
    def initialize_weights(self, num_features):
        #DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w

    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """
        # INSERT YOUR CODE HERE
        #raise Exception('Function not yet implemented!')

        # Computing the loss using the below formula
        # Loss = -(1/m)*sum( (y_i)*log(σ(wTx_i)) + (1-y_i)*log(1 - σ(wTx_i)))
        # m = number of examples and i for ith example

        loss = 0
        X = np.append(X, np.array([[1]]*X.shape[0]), axis=1)
        # for idx,example in enumerate(X):
        #     loss = loss + y[idx] * np.log(self.sigmoid(np.dot(example, self.w))) + (1 - y[idx]) * np.log(1 - self.sigmoid(np.dot(example, self.w)))
        # loss = -loss/ X.shape[0]

        loss =  -np.mean(y * np.log(self.sigmoid(np.dot(X, self.w))) + (1 - y) * np.log(1 - self.sigmoid(np.dot(X, self.w))))
        return loss
          
    def sigmoid(self, val):
        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """
        # INSERT YOUR CODE HERE
        # raise Exception('Function not yet implemented!')
        val = 1/(1+np.exp(-val))
        return val

    def gradient_ascent(self, w, X, y, lr):

        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """
        # INSERT YOUR CODE HERE
        #raise Exception('Function not yet implemented!')
        # gradient = x_j*(y-σ(wTX))
        return np.dot(X.T, y-self.sigmoid(np.dot(X, w)))

    def fit(self,X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """
        # INSERT YOUR CODE HERE
        # Appending a column of 1's to the train set X
        X = np.append(X, np.array([[1]]*X.shape[0]), axis=1)
        self.w =self.initialize_weights(X.shape[1])

        if(recompute):
            #Reinitialize the model weights
            self.w = self.initialize_weights(X.shape[1])

        for _ in range(iters):        
            # Calculate gradient ascent and Update weights
            # w = w + lr * g / size of X
            self.w += lr*(1 /  X.shape[0])*self.gradient_ascent(self.w,X,y,lr)
            
    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
         # INSERT YOUR CODE HERE
        # raise Exception('Function not yet implemented!')
        x = np.append(x, 1)
        y = self.sigmoid(np.dot(x, self.w))
        if y>0.5:
            return 1
        else:
            return 0




