# this script is for unzip .br file
# this function uzizp a .bz file and return the path of the file

import bz2
import json
import subprocess


def unzipbz2_json(file_path):
    # Path where you want to save the decompressed file
    # Change the extension to '.json'
    output_file_path = file_path[:-4] + '_decompressed.json'

    with bz2.open(file_path, 'rt', encoding='utf-8') as bz_file:
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            for line in bz_file:
                try:
                    # Load JSON data from each line
                    json_data = json.loads(line)
                    # Write JSON data to output file
                    json.dump(json_data, out_file)
                    out_file.write('\n')
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

    print(f"Decompressed file saved to {output_file_path}")
    return output_file_path

def unzip_bz2_exc(file_path):
    try:
        subprocess.run(['bzip2', '-d', file_path], check=True)
        print(f"File {file_path} has been successfully decompressed.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
     # Path to your .bz2 file
    bz2_file_path = '/Users/chenyujie/Desktop/RC_2010-01 copy.bz2'

    # Call the function
    unzip_bz2_exc(bz2_file_path)

