from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

test_input_data = {
     "person_age": 20,
    "person_income": 1000,
    "person_home_ownership": "OWN",
    "person_emp_length": 2,
    "loan_intent": "EDUCATION",
    "loan_grade": 'A',
    "loan_amnt": 200000,
    "loan_int_rate": 15.0,
    "loan_percent_income": 0.4,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 1
}


def test_response_api():
    response = client.get("/")
    assert response.status_code == 200, "There's something wrong with the API"
    assert response.json() == {"msg":"Hello","status":"success"}, "Incorrect response"
    

def test_predict_api():
    
    test_input_data = {
        "person_age": 20,
        "person_income": 1000,
        "person_home_ownership": "OWN",
        "person_emp_length": 2,
        "loan_intent": "EDUCATION",
        "loan_grade": 'A',
        "loan_amnt": 200000,
        "loan_int_rate": 15.0,
        "loan_percent_income": 0.4,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 1
    }
    response = client.post("/predict", json = test_input_data)
    print(response)
    assert response.status_code == 200, "There's something wrong with the predict API"

def test_error_predict_api():
    test_input_data = {
        "person_age": 20,
        "person_income": 1000,
        "person_home_ownership": "fail",
        "person_emp_length": 2,
        "loan_intent": "EDUCATION",
        "loan_grade": 'A',
        "loan_amnt": 200000,
        "loan_int_rate": 15.0,
        "loan_percent_income": 0.4,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 1
    }
    response = client.post("/predict", json = test_input_data)
    print(response)
    assert response.status_code == 400,response.detail