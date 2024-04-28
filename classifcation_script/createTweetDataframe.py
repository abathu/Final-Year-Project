# this script is for reading dataframe
# this version is for 2023 data , may need to modifiy for other
# example of json: [data, includes]
# data: metainfo I need
# includes :


import pandas as pd
import json
import bz2
### a function to remove all useless characters like url 
from classification import remove_urls_mentions_symbols_tels


def get_eng_text_from_json(file_path):
    # Create a list to hold the dictionaries
    data_list = []
    with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as file:
        for line in file:
            try:
                data = json.loads(line)
        
                if data['data'].get('lang') == 'en':  # Check if the language is English
                    # Add a dictionary to the list for each JSON object
                    text = data['data'].get('text', '')
                    data_list.append({'text':  text,  
                                      'cleared_text' : remove_urls_mentions_symbols_tels (text)
                                      , 'created_utc': data['data'].get('created_utc', '')})
                    # You can add other fields as needed
            except json.JSONDecodeError:
                continue  # Skip lines that are not valid JSON

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return df






def read_dataframe_from_json(dataframe_path):
    with open('your_file.json') as file:
        data = json.load(file)


def get_column_names_from_json(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)  # Load each line as a JSON object
            # Return column names from the first object
            return list(data.keys())


def get_first_value_from_json(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)  # Load each line as a separate JSON object
            return data  # Return the first JSON object

# this function return all text information that is writen in english in a array





if __name__ == "__main__":
    print("Hi")
    print((get_eng_text_from_json("/Volumes/backUp/2023/1/5/10/7.json.bz2")))
