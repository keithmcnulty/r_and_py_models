import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# spit data into train test
def split_data(df: pd.DataFrame, parameters: dict) -> dict:
    """
    Split and select data for modeling
    :param df: Pandas Dataframe
    :param parameters: split paramaters
    :return: Pandas Dataframe
    """
    X = df[parameters["input_cols"]]
    y = df[parameters["target_col"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=parameters["test_size"],
                                                        random_state=int(parameters["random_state"]))
    return dict(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)

# scale data
def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """
    Scale data for modelling
    :param X_train: Pandas DataFrame
    :param X_test: Pandas DataFrame
    :return: List of Pandas DataFrames
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    return dict(X_train_scaled = X_train_scaled, X_test_scaled = X_test_scaled)

# XGB train CV
def train_xgb_crossvalidated(
        X_train: pd.DataFrame, y_train:pd.DataFrame, parameters: dict
) -> XGBClassifier:
    """
    Train crossvalidated XGB Classifier
    :param X_train: Pandas DataFrame
    :param y_train: Pandas DataFrame
    :param parameters: Parameters for cross validation
    :return: Model of class KNNClassifier
    """
    param_dist = {'n_estimators': stats.randint(1, 100),
                  'learning_rate': stats.uniform(0.01, 0.6),
                  'subsample': parameters['subsample'],
                  'max_depth': parameters['xgb_max_depth'],
                  'colsample_bytree': parameters['colsample_bytree'],
                  'min_child_weight': parameters['xgb_min_child_weight']
                  }
    kfold = KFold(n_splits=int(parameters['k']),
                  shuffle=parameters['k_shuffle'],
                  random_state=int(parameters['random_state']))
    xgbmodel = XGBClassifier(use_label_encoder=False)
    xgb_clf = RandomizedSearchCV(xgbmodel, param_distributions=param_dist,
                                 n_iter=int(parameters['n_iter']), scoring=parameters['scoring'],
                                 error_score=parameters['error_score'], verbose=int(parameters['verbose']),
                                 n_jobs=int(parameters['n_jobs']), cv=kfold, 
                                 random_state=int(parameters['random_state']))
    xgb_clf.fit(X_train, y_train.values.ravel())
    return xgb_clf

# generate classification report
def generate_classification_report(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Generate classification report for model
    :param model: model object
    :param X_test: Pandas DataFrame
    :param y_test: Pandas Dataframe of test target values
    :return: Classification Report
    """
    y_pred = model.predict(X_test)
    return pd.DataFrame(classification_report(y_test, y_pred, output_dict = True)).transpose().drop('support', axis = 1)
