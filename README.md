<b>VLM ICL Reasoning </b>

## Datasets, Models, and Evaluation

### Datasets
Added ScienceQA Train, A-OKVQA Train and Val (The codebase has A-OKVQA which however does not include the rationales) in [Google Drive](https://drive.google.com/drive/folders/1hlN_NK1RnLSmYE0u2wG2KniAkXjvwxWJ?usp=share_link).

- {Dataset_name}_QCME: add rationale to the input
- {Dataset_name}_correct: only select the samples with correct prediction for the support set

### Change the data root
- vlmeval/smp/file.py
```
def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('/coc/pskynet4/chuang475/datasets/')  # change to your directory
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root
```

- If the original tsv files have the image_path column, change them to your directory

- vlmeval/dataset/image_mcq.py: change the added dataset location to your files
```
'ScienceQA_TRAIN': '/coc/pskynet4/chuang475/datasets/LMUData/ScienceQA_TRAIN.tsv',
```

- vlmeval/inference.py: here I did a correction for the roots since I moved the data but you can remove it
```
elif retriever is not None:
    for demo in retriever.find(idx, num_shots):
        # demo is a dict list, correct the image path
        for k in demo:
            if k['type'] == 'image' and '/nethome/chuang475/LMUData' in k['value']:
                k['value'] = k['value'].replace('/nethome/chuang475/LMUData', '/coc/pskynet4/chuang475/datasets/LMUData')
        demo_msgs += demo
```

### Hyper-Parameters
- --icl_rationale: enable VLM Reasoning ICL with consistent format (first feed the support set (e.g., image+question) to resoning models to get the reasonings, e.g.,
```
Demo input: image + question
Demo output: <SUMMARY> <CAPTION><REASONING><CONCLUSION>
Query input: image + question
Query output (generated): <SUMMARY> <CAPTION><REASONING><CONCLUSION>
)
```

- --multi_step_icl: multi-stage reasoning, don't use it for now
```
Stage 1
Demo input: image + question
Demo output: <SUMMARY> 
Query input: image + question
Query output (generated): <SUMMARY>

Stage 2
Demo input: image + question + <SUMMARY> 
Demo output: <CAPTION>
Query input: image + question + <SUMMARY>
Query output (generated): <CAPTION>

Stage 3
Demo input: image + question + <SUMMARY>  + <CAPTION>
Demo output: <REASONING> 
Query input: image + question + <SUMMARY>  + <CAPTION>
Query output (generated): <REASONING> 

Stage 4
Demo input: image + question + <SUMMARY>  + <CAPTION> + <REASONING> 
Demo output: <CONCLUSION> 
Query input: image + question + <SUMMARY>  + <CAPTION> + <REASONING> 
Query output (generated): <CONCLUSION>
```

### Model changes
Basically I changed generate_inner function for in-context learning and added call_inner function for JICES.
- vlmeval/vlm/qwen2_vl/model.py
- vlmeval/vlm/llama_vision.py