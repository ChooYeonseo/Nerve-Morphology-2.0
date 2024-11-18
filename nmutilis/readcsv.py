import pandas as pd
import pprint

raw_path = './data/sn_data_mod.csv'
helper_path = './data/helper_data.csv'
# Read the CSV file
raw_data = pd.read_csv(raw_path)
raw_data.set_index(raw_data.columns[0], inplace=True)
raw_dict = {col: raw_data[col].to_dict() for col in raw_data.columns}

helper_data = pd.read_csv(helper_path)
helper_data.set_index(helper_data.columns[0], inplace=True)
helper_dict = {col: helper_data[col].to_dict() for col in helper_data.columns}

def to_float_if_possible(value):
    try:
        return float(value)  # Attempt to convert to float
    except ValueError:
        return value  # If conversion fails, return the original value

for i in raw_dict:
    for key in raw_dict[i]:
        raw_dict[i][key] = to_float_if_possible(raw_dict[i][key])

for i in helper_dict:
    for key in helper_dict[i]:
        helper_dict[i][key] = to_float_if_possible(helper_dict[i][key])

pprint.pprint(helper_dict)