import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

def preprocess_ohe(data: pd.DataFrame, ohe,categories) -> pd.DataFrame:
    ohe_feat = ohe.transform(data[['loan_intent','person_home_ownership']])
    column_names = [cat for sublist in categories for cat in sublist]
    
    # create dataframe
    ohe_df = pd.DataFrame(ohe_feat, columns = column_names, index = data.index)

    final_df = pd.concat([data, ohe_df], axis = 1)

    final_df = final_df.drop(columns = ['loan_intent','person_home_ownership'])

    return final_df

def ohe_fit(data_train: pd.DataFrame, column: str,params):
    categories = [params['loan_intent_status'],params['person_home_ownership_status']]
    ohe = OneHotEncoder(categories = categories, sparse_output=False)
    
    ohe.fit(data_train[[column]])
    
    joblib.dump(ohe, params["dataset_dump_path"]["processed"] + "ohe_fix.pkl")
    
    return ohe