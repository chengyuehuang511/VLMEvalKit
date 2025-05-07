#!/bin/bash
cd /coc/testnvme/chuang475/projects/VLMEvalKit
name="vlmeval"
model="Qwen2.5-VL-3B-Instruct"
# model="LLaVA-CoT"
# model="Llama-3.2-11B-Vision-Instruct"
# model="VLM-R1"
# dataset="MathVista_MINI"
# dataset="EMMA"
# dataset="ScienceQA_TEST"
# query_dataset="ScienceQA_TEST_QCME"  # _wo_last
# dataset="A-OKVQA"
# dataset="MME_CoT_TEST"
# dataset="LogicVista"
# dataset="LogicVista_Rationale"  # _wo_last
# dataset="TextVQA_VAL"

# support_dataset="TextVQA_TRAIN"
# query_dataset="TextVQA_VAL"

support_dataset="ScienceQA_TRAIN"
# support_dataset="ScienceQA_TRAIN_correct"
# support_dataset="ScienceQA_TRAIN_QCME"
# support_dataset="ScienceQA_TRAIN_QCME_correct"
# query_dataset="ScienceQA_TEST"

# support_dataset="A-OKVQA_TRAIN"
# support_dataset="A-OKVQA_TRAIN_correct"
# support_dataset="A-OKVQA_TRAIN_QCME"
# support_dataset="A-OKVQA_TRAIN_QCME_correct"
query_dataset="A-OKVQA_VAL"
rag_method="jices"
num_shots=4

for support_dataset in "ScienceQA_TRAIN" "ScienceQA_TRAIN_correct" "ScienceQA_TRAIN_QCME" "ScienceQA_TRAIN_QCME_correct"
do
# for support_dataset in "A-OKVQA_TRAIN" "A-OKVQA_TRAIN_correct" "A-OKVQA_TRAIN_QCME" "A-OKVQA_TRAIN_QCME_correct"
    for model in "VLM-R1" "LLaVA-CoT" # "flamingov2" "Qwen2.5-VL-3B-Instruct" "VLM-R1" "Llama-3.2-11B-Vision-Instruct" "LLaVA-CoT"
    do
        for rag_method in "random" "jices"
        do
            job_name="${name}_$(date +%Y%m%d_%H%M%S)"
            output_dir="output/${model}/suport_${support_dataset}/query_${query_dataset}/${rag_method}/${job_name}"
            mkdir -p "$output_dir"
            sbatch --export "ALL,model=${model},support_dataset=${support_dataset},query_dataset=${query_dataset},output_dir=${output_dir},rag_method=${rag_method},num_shots=${num_shots}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" scripts/srun.sh
        done
    done
done