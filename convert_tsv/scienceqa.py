"/nethome/chuang475/LMUData/ScienceQA_TRAIN.tsv"

"""
"question"	"answer"	"hint"	"image"	"task"	"grade"	"subject"	"topic"	"category"	"skill"	"lecture"	"solution"	"split"	"A"	"B"	"C"	"index"	"D"	"E"
['image', 'question', 'choices', 'answer', 'hint', 'task', 'grade',
       'subject', 'topic', 'category', 'skill', 'lecture', 'solution']
"""

import os
import json
import pandas as pd
import numpy as np

import pandas as pd
# from vlmeval.smp import *

# df = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/scienceqa/train-00000-of-00001-1028f23e353fbe3e.parquet")
# print(df.keys())

# # change the column ['image', 'question', 'choices', 'answer', 'hint', 'task', 'grade', 'subject', 'topic', 'category', 'skill', 'lecture', 'solution']
# # to "question"	"answer"	"hint"	"image"	"task"	"grade"	"subject"	"topic"	"category"	"skill"	"lecture"	"solution"	"split"	"A"	"B"	"C"	"index"	"D"	"E"
# # the choices are a list of <= 5 items

# # Add the new columns for A, B, C, D, E (choices split into separate columns)
# choices_split = pd.DataFrame(df['choices'].to_list(), columns=['A', 'B', 'C', 'D', 'E'])
# df = pd.concat([df, choices_split], axis=1)

# # Add 'split' column if needed (you can fill it with a default value or logic)
# df['split'] = 'train'  # For example, default split can be 'train'

# # Add 'index' as an additional column (this will just create a new column with the index values)
# df['index'] = df.index

# # {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD... }
# df['image'] = df['image'].apply(lambda x: x['bytes'] if x is not None else x)

# # map 0,1,2,3,4 to A,B,C,D,E
# df['answer'] = df['answer'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'})

# # delete choices
# df = df.drop(columns=['choices'])

# # Print the updated columns
# print("Updated columns:", df.columns)

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
import csv

# def decode_base64_to_image(base64_string, target_size=-1):
#     image_data = base64.b64decode(base64_string)
#     image = Image.open(io.BytesIO(image_data))
#     if image.mode in ('RGBA', 'P', 'LA'):
#         image = image.convert('RGB')
#     if target_size > 0:
#         image.thumbnail((target_size, target_size))
#     return image

# def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
#     image = decode_base64_to_image(base64_string, target_size=target_size)
#     base_dir = osp.dirname(image_path)
#     if not osp.exists(base_dir):
#         os.makedirs(base_dir, exist_ok=True)
#     image.save(image_path)

# # df['image'] = df['image'].apply(lambda x: base64.b64encode(x).decode('utf-8') if x is not None else x)

# # # Show a sample of the DataFrame
# # print(df.head())
# # print(df.iloc[10])

# df = load("/nethome/chuang475/LMUData/ScienceQA_TEST.tsv")
# print(df.image[10])
# base64.b64encode(df.image[10])

# decode_base64_to_image_file(base64.b64encode(df.image[10]).decode('utf-8'), "/coc/testnvme/chuang475/projects/VLMEvalKit/try.jpg")
# df.to_csv("/nethome/chuang475/LMUData/ScienceQA_TRAIN.tsv", sep="\t", index=False, header=True) # , quoting=1, quotechar='"'

# # from datasets import load_dataset

# # # Load the ScienceQA dataset
# # dataset = load_dataset("derek-thomas/ScienceQA")

# # # Access the training split
# # train_data = dataset["train"]

# # print(train_data[0])

# # # Extract image links
# # image_links = [item["image"] for item in train_data]

# # # Display the first few image links
# # print(image_links[:5])

import pandas as pd
science = "/coc/pskynet4/chuang475/datasets/LMUData/ScienceQA_TEST.tsv"
science_df = pd.read_csv(science, sep="\t", encoding="utf-8")
data = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/scienceqa/test-00000-of-00001-f0e719df791966ff.parquet")
print(len(data))
# fileter out data with image is none
data = data[data['image'].notna()]
print(len(data))
print(len(science_df))

# import pandas as pd
science = "/coc/pskynet4/chuang475/datasets/LMUData/a-okvqa.tsv"
science_df = pd.read_csv(science, sep="\t", encoding="utf-8")
data = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/validation-00000-of-00001-b2bd0de231b6326a.parquet")

df_train = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/train-00000-of-00002-c1d24de3bacb5e0c.parquet")
df_1 = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/train-00001-of-00002-6b4f3abe2dc385d0.parquet")
df_train = pd.concat([df_train, df_1], axis=0).reset_index(drop=True)
print(len(df_train))
df_train = df_train[df_train['image'].notna()]
print(len(df_train))

# fileter out data with image is none
data = data[data['image'].notna()]
print(len(data))
print(len(science_df))

science = "/coc/pskynet4/chuang475/datasets/LMUData/ScienceQA_TRAIN.tsv"
science_df = pd.read_csv(science, sep="\t", encoding="utf-8")
data = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/scienceqa/train-00000-of-00001-1028f23e353fbe3e.parquet")
print(len(data))
# fileter out data with image is none
data = data[data['image'].notna()]
print(len(data))
print(len(science_df))