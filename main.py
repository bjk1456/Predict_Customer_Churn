
# Build Inputs and test code. 
import churn_library

from churn_library import *

if __name__ == "__main__":
    data_frame = import_data("./data/bank_data.csv")
    perform_eda(data_frame)

    encoder_helper(data_frame)
    X_train, X_test, y_train, y_test = perform_feature_engineering(data_frame)
    create_random_forest_model(X_train, y_train)
    create_logistic_regression_model(X_train, y_train)
    lrc_plot = save_lrc_plot(X_test, y_test)
    save_lrc_rfc_plot(X_test, y_test, lrc_plot)
    feature_importance_plot(X_test)
   