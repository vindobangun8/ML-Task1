from src.utils.helper import load_params, load_joblib
from src.utils.helper import split_num_cat, concat_data
from src.preprocessing.one_hot_encoder import preprocess_ohe
from src.preprocessing.custom_mapper import custom_label_encoder
import pandas as pd

params = load_params(param_dir = "config/params.yaml")


def preprocess_process(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    categories = [params['loan_intent_status'],params['person_home_ownership_status']]
    cat_data, num_data = split_num_cat(data = data, params = params)
    print("sini")
    ohe = load_joblib(path = "data/processed/ohe_model.pkl")
    
    cat_ohe_data = preprocess_ohe(data = cat_data, ohe = ohe, categories=categories)
    
    cat_final_data = custom_label_encoder(data = cat_ohe_data, params = params)
    
    final_data = concat_data(data_cat = cat_final_data, data_num = num_data)
    return final_data
    # return categories
