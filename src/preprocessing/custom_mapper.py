import pandas as pd

def custom_label_encoder(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    MAPPER_PERSON_VALUE = {
        "N": 0,
        "Y": 1
    }

    for col in params["label_encoder_columns"]:
        data[col] = data[col].replace(MAPPER_PERSON_VALUE)    
    
    #loan grade
    MAPPER_LOAN_GRADE = {
        "A": 7,
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1
    }
    
    data['loan_grade'] = data['loan_grade'].replace(MAPPER_LOAN_GRADE)

    return data