'''
Author: Ben Kelly
Date Created: 1 Jan 2022

This class is used to identify credit card customers 
that are most likely to churn.

To that end it consists of the following core modules:
-Data Import
-Data preparation
-Visualization
-Model training
-Module save and export
-Evaluation metrics and plot
'''

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(d_f):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    d_f['Churn'] = d_f['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20,10))
    d_f['Churn'].plot(kind='hist', figsize=(20, 10), fontsize=26
                    ).get_figure().savefig('./images/eda/churn.jpg')
    plt.figure(figsize=(20,10))
    d_f['Customer_Age'].plot(kind='hist', figsize=(20, 10), fontsize=26
        ).get_figure().savefig('./images/eda/customer_age.jpg')
    plt.figure(figsize=(20,10))
    d_f.Marital_Status.value_counts('normalize'
        ).plot(kind='bar').get_figure().savefig('./images/eda/martial_status.jpg')
    plt.figure(figsize=(20,10))
    sns.distplot(d_f['Total_Trans_Ct']).get_figure().savefig('./images/eda/total_trans_ct.jpg')
    plt.figure(figsize=(20,10))
    sns.heatmap(d_f.corr(), annot=False, cmap='Dark2_r', linewidths = 2
               ).get_figure().savefig('./images/eda/heat_map.jpg')

def encoder_helper(d_f):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            d_f: pandas dataframe

    output:
            d_f: pandas dataframe with new columns for
    '''
    # gender encoded column
    gender_lst = []
    gender_groups = d_f.groupby('Gender').mean()['Churn']

    for val in d_f['Gender']:
        gender_lst.append(gender_groups.loc[val])

    d_f['Gender_Churn'] = gender_lst
    #education encoded column
    edu_lst = []
    edu_groups = d_f.groupby('Education_Level').mean()['Churn']

    for val in d_f['Education_Level']:
        edu_lst.append(edu_groups.loc[val])

    d_f['Education_Level_Churn'] = edu_lst

    #marital encoded column
    marital_lst = []
    marital_groups = d_f.groupby('Marital_Status').mean()['Churn']

    for val in d_f['Marital_Status']:
        marital_lst.append(marital_groups.loc[val])

    d_f['Marital_Status_Churn'] = marital_lst

    #income encoded column
    income_lst = []
    income_groups = d_f.groupby('Income_Category').mean()['Churn']

    for val in d_f['Income_Category']:
        income_lst.append(income_groups.loc[val])

    d_f['Income_Category_Churn'] = income_lst

    #card encoded column
    card_lst = []
    card_groups = d_f.groupby('Card_Category').mean()['Churn']

    for val in d_f['Card_Category']:
        card_lst.append(card_groups.loc[val])

    d_f['Card_Category_Churn'] = card_lst
    return d_f

def get_keep_cols(d_f):
    '''
    helper function to cull the d_f of unwanted columns
    
    input:
            d_f: pandas dataframe

    output:
            d_f: pandas dataframe with culled columns
    '''
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = d_f[keep_cols]
    return X

def perform_feature_engineering(d_f):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = d_f['Churn']
    X = get_keep_cols(d_f)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def create_random_forest_model(X_train, y_train):
    '''
    Create a random forest model and save it to ./models/
    
    input:
            X_train: training data from x axis
            y_train: training data from y axis
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=get_param_grid(), cv=5)
    cv_rfc.fit(X_train, y_train)
    #y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    #y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

def get_param_grid():
    '''
    helper function to obtained the parameter grid for GridSearchCV

    output:
            param_grid: a grid of parameters for GridSearchCV
    '''
    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    return param_grid

def create_logistic_regression_model(X_train, y_train):
    '''
    Create a logistic regression model and save it to ./models/
    
    input:
            X_train: training data from x axis
            y_train: training data from y axis
    '''
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    #y_train_preds_lr = lrc.predict(X_train)
    #y_test_preds_lr = lrc.predict(X_test)
    joblib.dump(lrc, './models/logistic_model.pkl')

def save_lrc_plot(X_test, y_test):
    '''
    create a plot of the logistic regression model and save it to ./models/
    
    input:
            X_test: x testing data
            y_test: y testing data

    output:
            lrc_plot: a plot of the regression model
    '''
    lr_model = joblib.load('./models/logistic_model.pkl')
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.savefig("images/results/lrc_plot.pdf")
    return lrc_plot

def save_lrc_rfc_plot(X_test, y_test, lrc_plot):
    '''
    create a plot of the logistic regression and random forrest models and 
    save them to ./models/
    
    input:
            X_test: x testing data
            y_test: y testing data
            lrc_plot: the logistic regression model

    output:
            lrc_plot: a plot of the regression model
    '''
    rfc_model = joblib.load('./models/rfc_model.pkl')
    plt.figure(figsize=(15, 8))
    _ax = plt.gca()
    #rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=_ax, alpha=0.8)
    lrc_plot.plot(ax=_ax, alpha=0.8)

    plt.savefig("images/results/lrc_rfc_plot.pdf")

def save_tree_explainer(X_test):
    '''
    create a plot of the tree explainer model and save it to ./images/results/
    
    input:
            X_test: x testing data
    '''
    rfc_model = joblib.load('./models/rfc_model.pkl')
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    plt.savefig("images/results/tree_explainer.pdf")

def feature_importance_plot(X_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    rfc_model = joblib.load('./models/rfc_model.pkl')
    # Calculate feature importances
    importances = rfc_model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig("images/results/rfc_feature_importance.png")
