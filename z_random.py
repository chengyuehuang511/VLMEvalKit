# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("openflamingo/OpenFlamingo-9B-vitl-mpt7b")
# model = AutoModelForCausalLM.from_pretrained("openflamingo/OpenFlamingo-9B-vitl-mpt7b")  # openflamingo/OpenFlamingo-9B-vitl-mpt7b

import pandas as pd
from vlmeval.smp import *

# df = load("./outputs/VLM-R1/VLM-R1_ScienceQA_TRAIN_openai_result.xlsx")
possible_result_files = "outputs/VLM-R1/T20250505_G802c153f/VLM-R1_ScienceQA_TRAIN_QCME_openai_result.xlsx"
if osp.exists(possible_result_files):
    df = load(possible_result_files)
    df = load(possible_result_files)
print(df.head())