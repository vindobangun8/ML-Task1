from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import pandas as pd
from src.utils.helper import load_joblib, load_params
from src.data_pipeline.data_defense import data_defense_checker
from src.preprocessing.preprocess import preprocess_process


# init models and params
params = load_params(param_dir = "config/params.yaml")
best_model = load_joblib(path = params["model_dump_path"] + "decision_tree_best_model.pkl")

# create FastAPI object
app = FastAPI()

# init base model to define the data type
class APIData(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: object
    cb_person_cred_hist_length: int

# for root dir website, do this process
@app.get("/")
def root():
    return {
        "msg": "Hello",
        "status": "success"
    }


# service for predict ML model based on input data from user
@app.post("/predict")
def predict(data: APIData):
    # Convert input data to DataFrame
    df_data = pd.DataFrame([data.dict()])

    #Validate using data checker
    try:
        data_defense_checker(input_data=df_data, params=params)
    except AssertionError as ae:
        raise HTTPException(status_code=400, detail=f"Error Input: {str(ae)}")
    
    # If valid, preprocess the data
    df_data = preprocess_process(data=df_data, params=params)
    
    # Predict the input data
    y_pred = best_model.predict(df_data)

    if y_pred.item() is None:
        return {
            "res": "Failed API",
            "loan_status": None,
            "status_code": 500,
            "error_msg": "Prediction returned None."
        }
        
    return {
        "res": "Found API",
        "loan_status": y_pred.item(),
        "status_code": 200,
        "error_msg": ""
    }
