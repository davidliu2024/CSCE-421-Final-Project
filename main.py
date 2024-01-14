# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


from data import preprocess_x, preprocess_y, returnCombined, saveDF
from parser_1 import parse
from model import Model


def main():
    # args = parse()

    train_x = "./data/train_x.csv"
    train_y = "./data/train_y.csv"

    ###### Your Code Here #######
    # Add anything you want here

    ############################

    processed_x_train = preprocess_x(train_x)
    processed_y_train = preprocess_y(train_y)


    ###### Your Code Here #######
    # Add anything you want here

    ############################

    model = Model()  # you can add arguments as needed
    model.fit(processed_x_train, processed_y_train)

    
    ###### Your Code Here #######
    # Add anything you want here

    ############################

    test_x = "./data/test_x.csv"

    processed_x_test = preprocess_x(test_x)

    prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results
    
    test_y_df = returnCombined(test_x, prediction_probs)
    saveDF(test_y_df, filename="test_y.csv")
    model.saveModel(filename='logistic_regression_model.joblib')

    ########################


if __name__ == "__main__":
    main()
