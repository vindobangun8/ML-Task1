import pandas as pd


def data_defense_checker(input_data: pd.DataFrame, params: dict) -> None:
    try:
        print("===== Start Data Defense Checker =====")
        # check data types
        assert set(input_data[params["features"]].select_dtypes("object").columns.to_list()).issubset(set(params["object_columns"])), "an error occurs in object columns"
        assert set(input_data[params["features"]].select_dtypes("int64").columns.to_list()).issubset(set(params["int64_columns"])), "an error occurs in integer columns"
        assert set(input_data[params["features"]].select_dtypes("float64").columns.to_list()).issubset(set(params["float64_columns"])), "an error occurs in float columns"

        # check values
        assert set(input_data.cb_person_default_on_file).issubset(set(params["value_status"])), "an error occurs on cb_person_default_on_file column"
        assert set(input_data.loan_grade).issubset(set(params["loan_grade_status"])), "an error occurs on loan_grade_status column"
        assert set(input_data.loan_intent).issubset(set(params["loan_intent_status"])), "an error occurs on loan_intent column"
        assert set(input_data.person_home_ownership).issubset(set(params["person_home_ownership_status"])), "an error occurs on person_home_ownership column"
        
    except AssertionError as e:
        raise e

    finally:
        print("===== Finish Data Defense Checker =====")