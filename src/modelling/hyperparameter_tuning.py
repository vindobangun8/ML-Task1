from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, accuracy_score
import numpy as np
import pandas as pd
from src.utils.helper import load_joblib, dump_joblib


def hyperparam_process(model_path: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = load_joblib(path = model_path)
    
    PARAMS_DT = {
    "max_depth": [None,5,10,15,20,25,30,35,40],
    'min_samples_split': [2, 5, 10,20,50],
    'min_samples_leaf': [1, 2, 5,10,20],
    'criterion': ['gini', 'entropy'],
    "max_features":[None,'sqrt','log2']
    }
    
    k_folds = KFold(n_splits = 5)
    
    best_dt_random = RandomizedSearchCV(estimator = model,
                                       param_distributions = PARAMS_DT,
                                       cv = k_folds,
                                       verbose = 3)
    
    best_dt_random.fit(X_train,y_train)
    
    return best_dt_random.best_params_


def best_model_train(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    best_model = DecisionTreeClassifier(min_samples_split=2,min_samples_leaf=5,max_depth=10,criterion='entropy')
    
    best_model.fit(X_train, y_train)
    
    dump_joblib(best_model, params["model_dump_path"] + "best_model.pkl")
    
    return best_model


def predict_best_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    f_beta_dt_best = fbeta_score(y_test, y_pred, beta=0.5)
    acc_dt_best = accuracy_score(y_test, y_pred)
    print(f"accuracy score Decision Tree : {acc_dt_best}")
    print(f"f-beta score Decision Tree : {f_beta_dt_best}")
