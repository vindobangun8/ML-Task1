from sklearn.linear_model import LogisticRegression
from src.utils.helper import dump_joblib
from sklearn.metrics import fbeta_score, r2_score
import pandas as pd


def modeling_linreg(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)
    
    dump_joblib(logreg, params["model_dump_path"] + "vanilla_linreg_model.pkl")
    
    return logreg


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f"F_beta: {fbeta_score(y_valid, y_pred_dummy, beta=0.5)}")
    # print(f"R2: {r2_score(y_valid, y_pred_dummy)}")
