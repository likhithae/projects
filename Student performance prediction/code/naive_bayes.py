# naive_bayes.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 3 for CS6375: Machine Learning.
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
# 3. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. 
#
# 4. Make sure to save your model in a pickle file after you fit your Naive 
# Bayes algorithm.
#

import numpy as np
from collections import defaultdict
import pandas as pd
import nltk
import pprint
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pickle 
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression

class Simple_NB():
    """
    A class for fitting the classical Multinomial Naive Bayes model that is especially suitable
    for text classifcation problems. Calculates the priors of the classes and the likelihood of each word
    occurring given a class of documents.
    """
    def __init__(self):
        #Instance variables for the class.
        self.priors = defaultdict(dict)
        self.likelihood= defaultdict(dict)
        self.columns = None


    def partition(self, x):
        """
        Partition the column vector into subsets indexed by its unique values (v1, ... vk)

        Returns a dictionary of the form
        { v1: indices of y == v1,
        v2: indices of y == v2,
        ...
        vk: indices of y == vk }, where [v1, ... vk] are all the unique values in the vector z.
        """
        # INSERT YOUR CODE HERE
        partition_x = {}
        for idx, num in enumerate(x):
            if num in partition_x:
                partition_x[num].append(idx)
            else:
                partition_x[num] = [idx]

        return partition_x
        raise Exception('Function not yet implemented!')


    def fit(self, X, y, column_names, alpha=1):
        """
        Compute the priors P(y=k), and the class-conditional probability (likelihood param) of each feature 
        given the class=k. P(y=k) is the the prior probability of each class k. It can be calculated by computing 
        the percentage of samples belonging to each class. P(x_i|y=k) is the number of counts feature x_i occured 
        divided by the total frequency of terms in class y=k.
        
        The parameters after performing smooothing will be represented as follows.
            P(x_i|y=k) = (count(x_i|y=k) + alpha)/(count(x|y=k) + |V|*alpha) 
            where |V| is the vocabulary of text classification problem or
            the size of the feature set

        :param x: features
        :param y: labels
        :param alpha: smoothing parameter

        Compute the two class instance variable 
        :param self.priors: = Dictionary | self.priors[label]
        :param self.likelihood: = Dictionary | self.likelihood[label][feature]

        """
        #INSERT CODE HERE
        V = len(column_names)-1
        self.columns = column_names[:-1]
        y_partition = self.partition(y)
        y_labels = y_partition.keys() #Replace None

        # Prior probabilities
        for label in y_labels:
            self.priors[label] = len(y_partition[label])/len(y)

        top_words = {}
        
        #Tip: Add an extra key in your likelihood dictionary for unseen data. This will be used when testing sample texts
        # that contain words not present in feature set
        
        for label in y_labels:
            #Enter Code Here
            # Conditional probabilities
            for idx,feature in enumerate(self.columns):
                count = 0
                for i,value in enumerate(X[:,idx]):
                    if y[i] == label:
                        count += value
                # calculate P(x_i|y=k)
                self.likelihood[label][feature] = count            
        
        # INSERT YOUR CODE HERE
        for label in y_labels:
            list_prob = []
            total_count = sum(self.likelihood[label].values())
            # P(x_i|y=k)/total frequency of terms in class y=k
            for feature in self.columns:
                count = self.likelihood[label][feature]
                self.likelihood[label][feature] = (count + alpha) / (total_count + V* alpha)
            
            # extra key for unseen data
            self.likelihood[label]["__unseen__"] = alpha/(total_count+V*alpha)
            list_prob = sorted(self.likelihood[label].items(), key=lambda item:item[1],reverse=True)
            top_words[label] = [(x[0],np.log(x[1])) for x in list_prob]# if len(str(x[0])) > 5]
            top_words[label] = top_words[label][:3]

        print("Top three words that have the highest class-conditional likelihoods for both the “Dropout” and “Graduated” classes for our naive bayes model\n")
        print(top_words)

        #raise Exception('Function not yet implemented!')


    def predict_example(self, x, sample_text=False, return_likelihood=False):
        """
        Predicts the classification label for a single example x by computing the posterior probability
        for each class value, P(y=k|x) = P(x_i|y=k)*P(y=k).
        The predicted class will be the argmax of P(y=k|x) over all the different k's, 
        i.e. the class that gives the highest posterior probability
        NOTE: Converting the probabilities into log-space would help with any underflow errors.

        :param x: example to predict
        :return: predicted label for x
        :return: posterior log_likelihood of all classes if return_likelihood=True
        """

        if sample_text:
            #Convert list of words to a term frequency dictionary
            x = self.partition(x)

        # INSERT YOUR CODE HERE
        # Calculating posterior probability log(P(y=k|x)) = log(P(x_i|y=k)) + log(P(y=k)) 
        posterior_likelihood = {}
        for label in self.priors.keys():
            posterior_likelihood[label] =  np.log(self.priors[label])
            if sample_text:
                not_seen = 0
                for feature in x:
                    if feature in self.columns:
                        posterior_likelihood[label] +=  np.log(self.likelihood[label][feature])
                    else:
                        not_seen += 1
                if not_seen > 0:
                    posterior_likelihood[label] +=  np.log(self.likelihood[label]["__unseen__"])

            else:
                for idx,feature in enumerate(self.columns):
                    if x[idx] > 0:
                        posterior_likelihood[label] +=  np.log(self.likelihood[label][feature])
        
        # Predicted label = argmax of the posterior probabilities calculated
        predicted_label = 0
        max_probability = posterior_likelihood[0]
        for label in self.priors.keys():
            if posterior_likelihood[label] > max_probability:
                max_probability = posterior_likelihood[label]
                predicted_label = label

        if return_likelihood:
            return predicted_label,posterior_likelihood
        else:
            return predicted_label


        raise Exception('Function not yet implemented!')



    




    