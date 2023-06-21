# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
import graphviz
import statistics
import matplotlib.pyplot as plt


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
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

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE

    entropy = 0
    # Count the unique values of (v1,... vk) in vector y
    value, count = np.unique(y, return_counts=True)
    for p in count.astype('float')/len(y):
        if p != 0.0:
            entropy += -p * np.log2(p)
    return entropy
    raise Exception('Function not yet implemented!')

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    mutual_info = 0
    y = np.array(y)
    partition_x = partition(x)

    for value in partition_x:
        mutual_info += (len(partition_x[value]) /
                        len(x)) * entropy(y[x == value])

    mutual_info = entropy(y) - mutual_info

    return mutual_info
    raise Exception('Function not yet implemented!')

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    decision_tree = {}
	
    # The above mentioned 3 base cases
    if(entropy(y)== 0):
        return y[0]
    elif attribute_value_pairs != None and len(attribute_value_pairs) == 0:
        return statistics.mode(y)
    elif depth >= max_depth:
        return statistics.mode(y)
    else:	
        # Select the best attribute-value pair using INFORMATION GAIN as the splitting criterion
        all_mutual_info = []
        for (i, v) in attribute_value_pairs:
            all_mutual_info.append(mutual_information(np.array(x[:, i] == v).astype(int), y))
        all_mutual_info = np.array(all_mutual_info)
        (final_attr, final_value) = attribute_value_pairs[np.argmax(all_mutual_info)]

        # Partition the data set based on the values of the selected attribute before the next recursive call to ID3.
        # The selected attribute-value pair is removed from the list of attribute_value_pairs.
        partition_attr = partition(np.array(x[:, final_attr] == final_value).astype(int))
        # attribute_value_pairs.remove((final_attr, final_value))
        attribute_value_pairs_copy = []
        for i in attribute_value_pairs:
            if i != (final_attr, final_value):
                attribute_value_pairs_copy.append(i)

        attribute_value_pairs = attribute_value_pairs_copy

        # Store the tree as a nested dictionary, where each entry is of the form (attribute_index, attribute_value, True/False): subtree
        for value, idx in partition_attr.items():
            decision_tree[(final_attr, final_value, bool(value))] = id3(x.take(idx, axis=0), y.take(idx, axis=0), attribute_value_pairs=attribute_value_pairs, depth=depth + 1, max_depth=max_depth)

    return decision_tree

    raise Exception('Function not yet implemented!')    

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # Leaf node
    if isinstance(tree, np.int64) or isinstance(tree,int):  
        return tree
    for value, branch in tree.items():
        if value[2] == (x[value[0]] == value[1]):
            if type(branch) is not dict:
                label = branch
            else:
                label = predict_example(x, branch)

            return label
    
    return tree
    raise Exception('Function not yet implemented!')

def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))




    
    
  

