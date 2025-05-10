# "/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TRAIN_local.tsv"
# "/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TRAIN.tsv"
# "/coc/pskynet4/chuang475/datasets/LMUData/images/M3CoT_TRAIN"

# check the length of tsv of the number of files in the folder are the same
import pandas as pd
df1 = pd.read_csv("/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TRAIN_local.tsv", sep='\t')
df2 = pd.read_csv("/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TRAIN.tsv", sep='\t')
print(len(df1))
print(len(df2))
# number of files in the folder
import os
import glob
folder = "/coc/pskynet4/chuang475/datasets/LMUData/images/M3CoT_TRAIN"
files = glob.glob(os.path.join(folder, "*.jpg"))
print(len(files))