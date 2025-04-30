"""
--textvqa_image_dir_path "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/train_images/" \
--textvqa_train_questions_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_questions_vqa_format.json" \
--textvqa_train_annotations_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_annotations_vqa_format.json" \
--textvqa_test_questions_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/val_questions_vqa_format.json" \
--textvqa_test_annotations_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/val_annotations_vqa_format.json" \
"""

"/nethome/chuang475/LMUData/TextVQA_VAL_local.tsv"

"""
"question"	"answer"	"hint"	"image"	"task"	"grade"	"subject"	"topic"	"category"	"skill"	"lecture"	"solution"	"split"	"A"	"B"	"C"	"index"	"D"	"E"
"""

import os
import json
import pandas as pd
import numpy as np

import pandas as pd

df = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/test-00000-of-00001-f0e719df791966ff.parquet")
print(df['image'][10])

# train_questions_json_path = "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_questions_vqa_format.json"
# train_annotations_json_path = "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_annotations_vqa_format.json"
# output_path = "/nethome/chuang475/LMUData/TextVQA_TRAIN.tsv"

# with open(train_questions_json_path, 'r') as f:
#     train_questions = json.load(f)["questions"]
# with open(train_annotations_json_path, 'r') as f:
#     train_annotations = json.load(f)["annotations"]

# # order according to the question id
# train_questions = sorted(train_questions, key=lambda x: x["question_id"])
# train_annotations = sorted(train_annotations, key=lambda x: x["question_id"])

# # "answers" to list
# for i in range(len(train_annotations)):
#     train_annotations[i]["answers"] = [x["answer"] for x in train_annotations[i]["answers"]]

# train_questions = pd.DataFrame(train_questions)
# train_annotations = pd.DataFrame(train_annotations)
# train_questions = train_questions.rename(columns={"question_id": "index", "image_id": "image_path"})
# train_annotations = train_annotations.rename(columns={"question_id": "index", "answers": "answer"})

# # merge
# train_questions = pd.merge(train_questions, train_annotations, on="index")
# train_questions = train_questions[["index", "question", "answer", "image_path"]]
# train_questions["image_path"] = train_questions["image_path"].apply(lambda x: os.path.join("/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/train_images/", x))
# # add .jpg
# train_questions["image_path"] = train_questions["image_path"].apply(lambda x: x + ".jpg")

# # save
# print(train_questions)
# train_questions.to_csv(output_path, sep="\t", index=False, header=True, quoting=1, quotechar='"')

