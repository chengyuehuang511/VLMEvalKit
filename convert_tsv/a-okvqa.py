import os
import json
import pandas as pd
import numpy as np

import pandas as pd

reference = "/nethome/chuang475/LMUData/a-okvqa.tsv"
reference = pd.read_csv(reference, sep="\t", header=0)
print(reference.keys())
print(reference.iloc[10])

# df = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/train-00000-of-00002-c1d24de3bacb5e0c.parquet")
# df_1 = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/train-00001-of-00002-6b4f3abe2dc385d0.parquet")
# df = pd.concat([df, df_1], axis=0).reset_index(drop=True)
df = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/validation-00000-of-00001-b2bd0de231b6326a.parquet")
print(len(df))
print(df.index)
print(df.keys())
print(df.iloc[10])

# change the column ['image', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales']
# to ['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'answer', 'category', 'image', 'source', 'comment', 'split'] + 'solution'
# the choices are a list of <= 5 items

# Add the new columns for A, B, C, D (choices split into separate columns)
choices_split = pd.DataFrame(df['choices'].to_list(), columns=['A', 'B', 'C', 'D'])
print(choices_split.head())
df = pd.concat([df, choices_split], axis=1)

# Add 'split' column if needed (you can fill it with a default value or logic)
df['split'] = 'val'  # For example, default split can be 'train'

# Add 'index' as an additional column (this will just create a new column with the index values)
df['index'] = df.index

# {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD... }
df['image'] = df['image'].apply(lambda x: x['bytes'] if x is not None else x)

# map 0,1,2,3 to A,B,C,D
df['answer'] = df['correct_choice_idx'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
df['solution'] = df['rationales']

# delete choices
df = df.drop(columns=['choices', 'question_id', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales'])

# hint: null, category: ALL, source: A-OKVQA, comment: null
df['hint'] = None
df['source'] = 'A-OKVQA'
df['comment'] = None
df['category'] = 'ALL'

# Print the updated columns
print("Updated columns:", df.columns)

import os
import io
import pandas as pd
import numpy as np
import string
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
import sys

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)

df['image'] = df['image'].apply(lambda x: base64.b64encode(x).decode('utf-8') if x is not None else x)

# Show a sample of the DataFrame
print(df.head())
print(df.iloc[10])

# decode_base64_to_image_file(base64.b64encode(df.image[10]).decode('utf-8'), "/coc/testnvme/chuang475/projects/VLMEvalKit/try.jpg")
df.to_csv("/nethome/chuang475/LMUData/A-OKVQA_VAL.tsv", sep="\t", index=False, header=True) # , quoting=1, quotechar='"'