# for all of the tsv files in /coc/pskynet4/chuang475/datasets/LMUData, if they have image_path column, change /nethome/chuang475/LMUData to /coc/pskynet4/chuang475/datasets/LMUData
import os
import pandas as pd
from vlmeval.smp import *

def change_image_path_in_tsv_files(root_dir):
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tsv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Read the TSV file
                df = load(file_path)
                
                # Check if 'image_path' column exists
                if 'image_path' in df.columns and '/nethome/chuang475/LMUData' in df['image_path'].values[0]:
                    # Change the image path
                    df['image_path'] = df['image_path'].str.replace('/nethome/chuang475/LMUData', '/coc/pskynet4/chuang475/datasets/LMUData')
                    
                    # Save the modified DataFrame back to TSV
                    dump(df, file_path)
                    print(df['image_path'].head())
                    print(f"Updated image paths in {file_path}")
                else:
                    print(f"No 'image_path' column found in {file_path}")

if __name__ == "__main__":
    # Specify the root directory containing the TSV files
    root_directory = "/coc/pskynet4/chuang475/datasets/LMUData"
    
    # Call the function to change image paths in TSV files
    change_image_path_in_tsv_files(root_directory)