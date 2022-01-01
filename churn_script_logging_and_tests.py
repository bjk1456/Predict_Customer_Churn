import os
import logging
from os.path import exists

from churn_library import import_data, encoder_helper, perform_feature_engineering

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: The file has rows and columns")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        assert exists("./images/eda/churn.jpg")
        assert exists("./images/eda/customer_age.jpg")
        assert exists("./images/eda/heat_map.jpg")
        assert exists("./images/eda/martial_status.jpg")
        assert exists("./images/eda/total_trans_ct.jpg")
        logging.info("Testing test_eda: All figures are present")
    except AssertionError as err:
        logging.error("Testing test_eda: Missing a figure")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df = encoder_helper(df)
    try:
        assert len(df['Gender']) > 2
        assert len(df['Education_Level']) > 2
        assert len(df['Marital_Status']) > 2
        assert len(df['Income_Category']) > 2
        assert len(df['Card_Category']) > 2
        logging.info("Testing test_encoder_helper: encoding succeded ")
    except AssertionError as err:
        logging.error("Testing test_eda: Missing a figure")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = encoder_helper(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    try:
        assert len(X_train) > 2
        assert len(X_test) > 2
        assert len(y_train) > 2
        assert len(y_test) > 2
        logging.info("Testing test_perform_feature_engineering: data successfully trained")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Missing training and/or testing data")
        raise err

if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
   