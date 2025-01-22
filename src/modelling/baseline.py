from sklearn.dummy import DummyClassifier
from src.utils.helper import dump_joblib # type: ignore
from sklearn.metrics import  accuracy_score,fbeta_score
import pandas as pd


def modeling_baseline(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    dummy_regr = DummyClassifier(strategy = "most_frequent")

    dummy_regr.fit(X_train, y_train)
    
    dump_joblib(dummy_regr, params["model_dump_path"] + "baseline_model.pkl")
    
    return dummy_regr


def predict_baseline(model, X_valid, y_valid):
    y_pred_dummy = model.predict(X_valid)
    
    print(f"MSE: {accuracy_score(y_valid,y_pred_dummy)}")
    print(f"R2: {fbeta_score(y_valid, y_pred_dummy, beta=3)}")
