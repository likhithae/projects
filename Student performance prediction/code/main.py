

#Libraries to Install

# pip install python-javabridge
# pip install python-weka-wrapper3
# pip install seaborn
# pip install skfeature-chappers

import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import graphviz

# sns.set_theme(style="darkgrid")
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})

# Importing sklearn models and metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from skfeature.function.similarity_based import fisher_score
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO

# Importing Models
import logistic_regression as lr
import decision_tree as dt
import naive_bayes as nb
import knn as KNN

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def confusionMatrix1(yTrue,yPred):
    true_pos,false_pos = 0,0
    true_neg,false_neg = 0,0
    for idx in range(len(yTrue)):
        if yTrue[idx]==1 and yPred[idx]==1:
            true_pos += 1
        if yTrue[idx]==0 and yPred[idx]==1:
            false_pos += 1
        if yTrue[idx]==0 and yPred[idx]==0:
            true_neg += 1
        if yTrue[idx]==1 and yPred[idx]==0:
            false_neg += 1

    confusion_matrix = np.array([true_pos,false_neg,false_pos,true_neg]).reshape(2,2)		
    print("\t",confusion_matrix[0][0],"\t",confusion_matrix[0][1])
    print("\t",confusion_matrix[1][0],"\t",confusion_matrix[1][1])
    
    return confusion_matrix

def computeError(yTrue, yPred):
    # error rate = (1/n) * sum(y_true!=y_pred)
    error = 0
    for idx,y in enumerate(yTrue):
        if yTrue[idx] != yPred[idx]:
            error += 1
    return error/len(yTrue)

def computeAccuracy(yTrue, yPred):
    # error rate = (1/n) * sum(y_true!=y_pred)
    # accuracy = 1 - error
    error = 0
    for i in range(len(yTrue)):
        if yTrue[i] != yPred[i]:
            error += 1
        
    error = error/len(yTrue)
    return round(1-error,5)
    raise Exception('Function not yet implemented!')

def computePrecision(yTrue, yPred):
    # precision = tp/tp+fp
    precision = 0
    tp = 0
    fp = 0

    for i in range(len(yTrue)):
        if yTrue[i] == 1 and yPred[i] == 1:
            tp += 1
        if yTrue[i] == 0 and yPred[i] == 1:
            fp += 1

    if tp == 0 and fp ==0:
        precision = 0
    else:    
        precision = tp/(tp+fp)
    
    return round(precision,5)

def computeRecall(yTrue, yPred):
    # recall = tp/tp+fn = tp/total positives
    recall = 0
    tp = 0
    total_pos = 0

    for i in range(len(yTrue)):
        if yTrue[i] == 1:
            total_pos += 1
            if yPred[i] == 1:
                tp += 1

    recall = tp/total_pos
    return round(recall,5)

def computeF1(yTrue, yPred):
    # f1 = 2 * (P*R)/(P+R)
    precision = computePrecision(yTrue, yPred)
    recall = computeRecall(yTrue, yPred)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision + recall)

    return round(f1,5)
    
def computeROC(yTrue, yPred):
    pass 

def modelResults(yTrain, yTest, yVal, predictionTrain, predictionTest, predictionVal, model):
    print("\nModel - ", model)
    print("Training error - ", round(computeError(yTrain,predictionTrain)*100,2),"%")
    if predictionVal != '':
        print("Validation error - ", round(computeError(yVal,predictionVal)*100,2),"%")
    print("Test error - ", round(computeError(yTest,predictionTest)*100,2),"%")
    print("Accuracy - ", round(computeAccuracy(yTest,predictionTest)*100,2),"%")
    print("Precision - ", round(computePrecision(yTest,predictionTest)*100,2),"%")
    print("Recall - ", round(computeRecall(yTest,predictionTest)*100,2),"%")
    print("F1 Score - ", round(computeF1(yTest,predictionTest)*100,2),"%")
    print('Confusion Matrix')
    confusionMatrix1(yTest,predictionTest)
    # fpr, tpr, _ = metrics.roc_curve(yTest, predictionTest)
    # auc = metrics.roc_auc_score(yTest, predictionTest)

    # #create ROC curve
    # plt.plot(fpr,tpr,label="AUC="+str(auc))
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc=4)
    # plt.show()

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    PrecisionRecallDisplay.from_predictions(yTest, predictionTest)
    plt.title(f"Precision-Recall Curve %s"%model)
    RocCurveDisplay.from_predictions(yTest, predictionTest)
    plt.title(f"ROC Curve %s"%model)
    plt.show()

def scikitModel(xTrain, xTest, yTrain, yTest, model):
    scikitModel = model
    scikitModel.fit(xTrain,yTrain)
    predictionTrain = scikitModel.predict(xTrain)
    predictionTest = scikitModel.predict(xTest)

    return predictionTrain,predictionTest

def logisticRegression(xTrain, xTest, yTrain, yTest, xVal, yVal):
    logisticRegression = lr.SimpleLogisiticRegression()
    bestLr = 0
    bestItr = 0
    valErrMin = 1000
    results = []
    totalValError = 0

    for iter in [10,100,1000,10000]:
        for a in [0.01,0.1,0.33]:
            yPred = []

            # Training the model for different parameters and predicting the labels for the validation set
            logisticRegression.fit(xTrain, yTrain,a,iter)
            for idx,example in enumerate(xVal):
                yPred.append(logisticRegression.predict_example(logisticRegression.w,example))
            yPred = np.array(yPred)
            # Calculating the error for the Validation label predictions
            valErr = computeError(yVal,yPred)
            # Choosing the best parameter based on the Validation set error
            if valErr<=valErrMin:
                valErrMin = valErr 
                bestLr = a
                bestItr = iter
            
            totalValError += valErr
            results.append([iter,a,round(valErr,5)])
            
    print("\n Logistic Regression")
    print("Results for Validation Error for different number of iterations and learning rates")
    print ("{:<8} {:<15} {:<12}".format('Iterations',' Learning rate','Validation error'))
    for res in results:
        print ("{:<14} {:<15} {:<11}".format(res[0], res[1], res[2]))

    # Retrain Logistic Regression on the best parameters
    print("\nThe best parameters for logistic regression that gives minimum error on the validation data is")
    print("Iterations - ",bestItr,"   Learning rate - ",bestLr,"   Validation error - ", valErr)
    logisticRegression.fit(xTrain, yTrain,bestLr,bestItr)
    predictionTrain = [logisticRegression.predict_example(logisticRegression.w,example) for idx,example in enumerate(xTrain)]
    predictionVal = [logisticRegression.predict_example(logisticRegression.w,example) for idx,example in enumerate(xVal)]
    predictionTest = [logisticRegression.predict_example(logisticRegression.w,example) for idx,example in enumerate(xTest)]
    
    return predictionTrain, predictionTest, predictionVal

def decisionTree(xTrain, xTest, yTrain, yTest, xVal, yVal):
    attr_value_pairs = []
    for i in range(xTrain.shape[1]):
        for v in np.unique(xTrain[:, i]):
            attr_value_pairs.append((i,v))

    bestDepth = 0
    valErrMin = 1000
    results = []

    for depth in range(1,10):
        decision_tree = dt.id3(xTrain, yTrain,attribute_value_pairs=attr_value_pairs, depth=0, max_depth=depth)    
        yPred = [dt.predict_example(x, decision_tree) for x in xVal]
        yPred = np.array(yPred)
        valErr = computeError(yVal,yPred)
        # Choosing the best parameter based on the Validation set error
        if valErr<=valErrMin:
            valErrMin = valErr 
            bestDepth = depth
        
        results.append([depth,round(valErr,5)])
            
    print("\n Decision Tree")
    print("Results for Validation Error for different depths")
    print ("{:<8} {:<15}".format('Depth','Validation error'))
    for res in results:
        print ("{:<14} {:<15}".format(res[0], res[1]))

    # Retrain decision tree on the best parameters 
    print("\nThe best parameters for decision tree that gives minimum error on the validation data is")
    print("Depth - ",bestDepth,"   Validation error - ", valErr)
    
    decision_tree = dt.id3(xTrain, yTrain,attribute_value_pairs=attr_value_pairs, depth=0, max_depth=bestDepth)
    dt.visualize(decision_tree, depth=0)
    # graph = graphviz.Source(tree.export_graphviz(decision_tree, out_file=None)) 
    # graph.render("graph_spect")
    predictionTrain = [dt.predict_example(x, decision_tree) for x in xTrain]
    predictionVal = [dt.predict_example(x, decision_tree) for x in xVal]
    predictionTest = [dt.predict_example(x, decision_tree) for x in xTest]
    
    return predictionTrain, predictionTest, predictionVal

def naiveBayes(xTrain, xTest, yTrain, yTest, xVal, yVal, xColumns):
    naiveBayes = nb.Simple_NB()
    naiveBayes.fit(xTrain, yTrain, column_names=xColumns, alpha=1)
    
    predictionTrain = [naiveBayes.predict_example(x) for x in xTrain]
    predictionVal = [naiveBayes.predict_example(x) for x in xVal]
    predictionTest = [naiveBayes.predict_example(x) for x in xTest]

    return predictionTrain, predictionTest, predictionVal

def knn(xTrain, xTest, yTrain, yTest, xVal, yVal):
    knn1 = KNN.knn()
    bestK = 0
    valErrMin = 1000
    results = []

    for k in range(1,10):
        yPred = knn1.predict_example(xTrain, yTrain, xTest, k)
        yPred = np.array(yPred)
        valErr = computeError(yVal,yPred)
        # Choosing the best parameter based on the Validation set error
        if valErr<=valErrMin:
            valErrMin = valErr 
            bestK = k
        
        results.append([k,round(valErr,5)])
            
    print("\n KNN")
    print("Results for Validation Error for different values of K")
    print ("{:<8} {:<15}".format('K','Validation error'))
    for res in results:
        print ("{:<14} {:<15}".format(res[0], res[1]))

    # Retrain decision tree on the best parameters 
    print("\nThe best parameters for KNN that gives minimum error on the validation data is")
    print("K - ",bestK,"   Validation error - ", valErr)
    
    predictionTrain = knn1.predict_example(xTrain, yTrain, xTrain, bestK)
    predictionVal = knn1.predict_example(xTrain, yTrain, xVal, bestK)
    predictionTest = knn1.predict_example(xTrain, yTrain, xTest, bestK)
    
    return predictionTrain, predictionTest, predictionVal

if __name__ == '__main__':
    # Load the Complete data
    try:
        data = pd.read_csv("data.csv")
    except:
        data = pd.read_csv("dataset.csv")
    
        # Remove the rows where the target class is enrolled.
        data = data[data.Target != 'Enrolled']
        
        # Updating the values of the target variables (graduate/dropout) to integers (1/0)
        data['Target'] = data['Target'].replace({'Graduate': 1})
        data['Target'] = data['Target'].replace({'Dropout': 0})
        
        data.to_csv("data.csv", index=False)


    # removeCols = ['Curricular units 1st sem (enrolled)','Curricular units 2nd sem (enrolled)']
    # Store headers as column names
    dataCsv = pd.read_csv('data.csv')
    # for feature in removeCols:
    #     dataCsv = dataCsv.drop(feature, axis=1)
    xColumns = list(dataCsv.columns)[:-1]
    
    # Read the dataset as a dataframe for further analysis 
    dataset = np.genfromtxt('data.csv', missing_values=0, skip_header=0, delimiter=',',dtype=float)
    # Remove the headers 
    dataset = dataset[1:]

    print("Number of rows in the dataset - ", dataset.shape[0])
    print("Number of columns in the dataset - " ,dataset.shape[1],"\n")
    X = dataset[:, :-1]
    Y = dataset[:, -1]

    
    # Heatmap
    plt.figure(figsize=(12,12))
    corrmat = dataCsv.corr()
    top_corr_features = corrmat.index
    g=sns.heatmap(dataCsv[top_corr_features].corr(),cmap="RdYlGn")
    # print(top_corr_features)
    plt.subplots_adjust(left=0.25,bottom=0.31)
    plt.title("Heat map")
    plt.show()
    cor_target = abs(corrmat["Target"])
    cor_target.nlargest(20).plot(kind='barh')
    plt.title("Top 20 Highly correleated features with the target variable")
    plt.subplots_adjust(left=0.35)
    plt.show()

    # cor_target = cor_target[cor_target>0.2]
    sortedSeries = cor_target.sort_values(ascending=False)
    relevant_features = []
    for a,b in sortedSeries.items():
        relevant_features.append(a)
    relevant_features = relevant_features[1:21]
    # relevant_features.append("Target")
    print("\nImportant features selected based on the correlation with the Target Variable")
    print(relevant_features)

    # Fisher score
    # ranks = fisher_score.fisher_score(X,Y)
    # feat_importances = pd.Series(ranks, index=xColumns)
    # feat_importances.nlargest(15).plot(kind='barh')
    # plt.subplots_adjust(left=0.35)
    # plt.show()
    # sortedSeries = feat_importances.sort_values(ascending=False)
    # relevant_features = []
    # for a,b in sortedSeries.items():
    #     relevant_features.append(a)
    # relevant_features = relevant_features[:15]
    # print(relevant_features)

    # Taking the data with only the important features
    selectedCols = []
    for idx, val in enumerate(xColumns):
        if xColumns[idx] in relevant_features:
            selectedCols.append(idx)

    xColumns = relevant_features
    X = X[:, selectedCols]

    # Exploratory Data Analysis
    numericalFeatures = []
    categoricalFeatures = []
    for feature in relevant_features:
        if len(set(dataCsv[feature])) == 2:
            categoricalFeatures.append(feature)
        else:
            numericalFeatures.append(feature)

    for feature in numericalFeatures:
        sns.jointplot(x=dataCsv[feature], y=dataCsv['Target'])
        plt.show()

    sns.countplot(x ='Target', data = dataCsv)
    plt.title("Count plot for the Target variable")
    plt.show()

    dataCsv[numericalFeatures[:4]].hist(figsize=(15, 6), layout=(2, 2))
    plt.show()
    dataCsv[numericalFeatures[4:8]].hist(figsize=(15, 6), layout=(2, 2))
    plt.show()
    dataCsv[numericalFeatures[8:12]].hist(figsize=(15, 6), layout=(2, 2))
    plt.show()
    dataCsv[numericalFeatures[12:]].hist(figsize=(15, 6), layout=(2, 2))
    plt.show()

    for col in numericalFeatures:
        sns.boxplot(y = dataCsv['Target'].astype('category'), x = col, data=dataCsv)
        plt.show()
    

    # Divide the data into train and test datasets
    # Training - 80%, Test - 10%, Validation - 10%
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)
    xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.125, random_state=1) # 0.125 x 0.8 = 0.1

    # Implementing Own models
    # Logistic regression
    result = logisticRegression(xTrain, xTest, yTrain, yTest, xVal, yVal)
    modelResults(yTrain, yTest, yVal, result[0], result[1], result[2],  "Logistic regression")    
    
    # Decision tree
    result = decisionTree(xTrain, xTest, yTrain, yTest, xVal, yVal)
    modelResults(yTrain, yTest, yVal, result[0], result[1], result[2], "Decision tree")    
    
    # Naive Bayes
    result = naiveBayes(xTrain, xTest, yTrain, yTest, xVal, yVal, xColumns)
    modelResults(yTrain, yTest, yVal, result[0], result[1], result[2], "Naive Bayes")    
    
    # KNN
    result = knn(xTrain, xTest, yTrain, yTest, xVal, yVal)
    modelResults(yTrain, yTest, yVal, result[0], result[1], result[2], "KNN")    

    
    # Implementing Scikit models
    print("\n------------------------------------------------------------\n")
    print("\n Implementing Scikit Models")

    # Logistic regression
    result = scikitModel(xTrain, xTest, yTrain, yTest, LogisticRegression())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Logistic regression")    
    
    # Decision tree
    result = scikitModel(xTrain, xTest, yTrain, yTest, DecisionTreeClassifier())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Decision tree")    

    # Naive Bayes
    # Error for negative values
    result = scikitModel(xTrain, xTest, yTrain, yTest, MultinomialNB())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Multinomial Naive Bayes")    

    result = scikitModel(xTrain, xTest, yTrain, yTest, GaussianNB())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Gaussian Naive Bayes")    

    result = scikitModel(xTrain, xTest, yTrain, yTest, BernoulliNB())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Bernoulli Naive Bayes")   

    # kNN
    result = scikitModel(xTrain, xTest, yTrain, yTest, NearestCentroid())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's KNN")   
    
    # SVM
    result = scikitModel(xTrain, xTest, yTrain, yTest, svm.SVC())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's SVM SVC")   

    result = scikitModel(xTrain, xTest, yTrain, yTest, svm.LinearSVC())
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's SVM LinearSVC")   

    # Gradient Boosting Classifier
    result = scikitModel(xTrain, xTest, yTrain, yTest, GradientBoostingClassifier(n_estimators=100))
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Stochastic Gradient Boosting Classifier")   

    # Neural network
    result = scikitModel(xTrain, xTest, yTrain, yTest, MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))
    modelResults(yTrain, yTest, yVal, result[0], result[1], '', "scikit-learn's Neural network")   
