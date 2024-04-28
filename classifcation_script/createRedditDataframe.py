# this script is for reading dataframe from raddit dataset files
# this version is for 2023 data , may need to modifiy for other
# example of json: [data, includes]
# data: metainfo I need
# includes :
import json
from datetime import datetime
import pandas as pd


def read_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_line = json.loads(line)

                # Convert the 'created_utc' timestamp to a human-readable date
                date = datetime.utcfromtimestamp(
                    int(json_line['created_utc'])).strftime('%Y-%m-%d')

                # Store only 'created_utc' (as date) and 'body'
                data.append({
                    'created_utc': date,
                    'body': json_line['body']
                })
            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")
                continue
            except KeyError as e:
                print(f"Missing key {e} in JSON line: {line}")
                continue
    return data


if __name__ == '__main__':
    # Path to your JSON file
    file_path = '/Volumes/backUp/redit/reddit_data/2007/RC_2007-10'

    # Read the file
    data = read_json_file(file_path)
    df = pd.DataFrame(data)
    print(df.head(6))
    print(len(df))


    # pass to classification method 

    # save the results

    
