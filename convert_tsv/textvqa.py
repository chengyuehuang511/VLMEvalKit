"""
--textvqa_image_dir_path "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/train_images/" \
--textvqa_train_questions_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_questions_vqa_format.json" \
--textvqa_train_annotations_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_annotations_vqa_format.json" \
--textvqa_test_questions_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/val_questions_vqa_format.json" \
--textvqa_test_annotations_json_path "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/val_annotations_vqa_format.json" \
"""

"/nethome/chuang475/LMUData/TextVQA_VAL_local.tsv"

"""
"index"	"question"	"answer"	"image_path"
"34602"	"what is the brand of this camera?"	"['nous les gosses', 'dakota', 'clos culombu', 'dakota digital', 'dakota', 'dakota', 'dakota digital', 'dakota digital', 'dakota', 'dakota']"	"/nethome/chuang475/LMUData/images/TextVQA_VAL/34602.jpg"

{"question_id": 0, "image_id": "0054c91397f2fe05", "question_type": "none of the above", "answers": [{"answer": "nokia", "answer_confidence": "yes", "answer_id": 1}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 2}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 3}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 4}, {"answer": "toshiba", "answer_confidence": "yes", "answer_id": 5}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 6}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 7}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 8}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 9}, {"answer": "nokia", "answer_confidence": "yes", "answer_id": 10}], "multiple_choice_answer": "nokia"}
{"question": "what is the brand of phone?", "image_id": "0054c91397f2fe05", "question_id": 0}
"""

import os
import json
import pandas as pd
import numpy as np
train_questions_json_path = "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_questions_vqa_format.json"
train_annotations_json_path = "/coc/testnvme/chuang475/projects/VQA-ICL/data/textvqa/train_annotations_vqa_format.json"
output_path = "/nethome/chuang475/LMUData/TextVQA_TRAIN.tsv"

with open(train_questions_json_path, 'r') as f:
    train_questions = json.load(f)["questions"]
with open(train_annotations_json_path, 'r') as f:
    train_annotations = json.load(f)["annotations"]

# order according to the question id
train_questions = sorted(train_questions, key=lambda x: x["question_id"])
train_annotations = sorted(train_annotations, key=lambda x: x["question_id"])

# "answers" to list
for i in range(len(train_annotations)):
    train_annotations[i]["answers"] = [x["answer"] for x in train_annotations[i]["answers"]]

train_questions = pd.DataFrame(train_questions)
train_annotations = pd.DataFrame(train_annotations)
train_questions = train_questions.rename(columns={"question_id": "index", "image_id": "image_path"})
train_annotations = train_annotations.rename(columns={"question_id": "index", "answers": "answer"})

# merge
train_questions = pd.merge(train_questions, train_annotations, on="index")
train_questions = train_questions[["index", "question", "answer", "image_path"]]
train_questions["image_path"] = train_questions["image_path"].apply(lambda x: os.path.join("/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/train_images/", x))
# add .jpg
train_questions["image_path"] = train_questions["image_path"].apply(lambda x: x + ".jpg")

# save
print(train_questions)
train_questions.to_csv(output_path, sep="\t", index=False, header=True, quoting=1, quotechar='"')

