from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import joblib

class Model():
    def __init__(self):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        self.n_folds = 10
        self.kFold_model = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
        self.class_weights = {0: 1, 1: 11}
        self.model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, class_weight=self.class_weights)


        ########################################################################

    def fit(self, trainX, trainY):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        roc_scores = []
        for train, test in self.kFold_model.split(trainX, trainY):
            x_train, x_test = trainX[train], trainX[test]
            y_train, y_test = trainY[train], trainY[test]
            
            self.model.fit(x_train, y_train)
            y_proba = self.model.predict_proba(x_test)[:,1]
            y_pred = (y_proba >= 0.5).astype(int)


            roc_scores.append(roc_auc_score(y_test, y_pred))
        return np.mean(roc_scores)

        ########################################################################

    def predict_proba(self, test_x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x

        ########################################################################
        return self.model.predict_proba(test_x)[:,1]
    
    def saveModel(self, filename):
        joblib.dump(self.model, filename=filename)