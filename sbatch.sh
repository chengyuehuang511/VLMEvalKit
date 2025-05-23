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

# query_dataset="OK-VQA_VAL"

# support_dataset="ScienceQA_TRAIN"
# support_dataset="ScienceQA_TRAIN_correct"
# support_dataset="ScienceQA_TRAIN_QCME"
# support_dataset="ScienceQA_TRAIN_QCME_correct"
# query_dataset="ScienceQA_TEST"

# support_dataset="A-OKVQA_TRAIN"
# support_dataset="A-OKVQA_TRAIN_correct"
# support_dataset="A-OKVQA_TRAIN_QCME"
# support_dataset="A-OKVQA_TRAIN_QCME_correct"
# query_dataset="A-OKVQA_VAL"

# support_dataset="M3CoT_TRAIN"
# support_dataset="M3CoT_TRAIN_correct"
# support_dataset="M3CoT_TRAIN_QCME"
# support_dataset="M3CoT_TRAIN_QCME_correct"
# query_dataset="M3CoT_TEST"

# for support_dataset in "ScienceQA_TRAIN" #"ScienceQA_TRAIN_QCME_correct" # "ScienceQA_TRAIN" # "ScienceQA_TRAIN_QCME" "ScienceQA_TRAIN_QCME_correct"
# for support_dataset in "A-OKVQA_TRAIN" #"A-OKVQA_TRAIN_QCME_correct" #"A-OKVQA_TRAIN_correct" "A-OKVQA_TRAIN_QCME" "A-OKVQA_TRAIN_QCME_correct"
for support_dataset in "OK-VQA_TRAIN"
do
    for query_dataset in "OK-VQA_VAL"
    # for support_dataset in "OK-VQA_TRAIN"
    # for support_dataset in "M3CoT_TRAIN" #"M3CoT_TRAIN" "M3CoT_TRAIN_QCME" #"M3CoT_TRAIN" #"M3CoT_TRAIN_correct" "M3CoT_TRAIN_QCME" "M3CoT_TRAIN_QCME_correct"
    do
        # for model in "Qwen2.5-VL-3B-Instruct" #"InternVL2_5-8B" "InternVL2_5-8B" "Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-7B-Instruct" "Llama-3.2-11B-Vision-Instruct"
        for model in "GeminiFlash2-0-thinking" #"GeminiFlash2-5-thinking" "GeminiFlash1-5" 
        # for model in "InternVL2_5-4B-MPO" "InternVL2_5-8B-MPO" #"VLM-R1" "VL-Rethinker-7B" "LLaVA-CoT" 
        #"VL-Rethinker-7B" #"InternVL2_5-8B-MPO" #"InternVL2_5-4B" "InternVL2_5-4B-MPO" #"Llama-3.2-11B-Vision-Instruct" # "flamingov2" "Qwen2.5-VL-3B-Instruct" "VLM-R1" "Llama-3.2-11B-Vision-Instruct" "LLaVA-CoT" "Kimi-VL-A3B-Instruct" "Kimi-VL-A3B-Thinking"
        do
            for rag_method in "random"
            do
                job_name="${name}_$(date +%Y%m%d_%H%M%S)"
                output_dir="output/${model}/suport_${support_dataset}/query_${query_dataset}/${rag_method}/${job_name}"
                mkdir -p "$output_dir"
                sbatch --export "ALL,model=${model},support_dataset=${support_dataset},query_dataset=${query_dataset},output_dir=${output_dir},rag_method=${rag_method},num_shots=${num_shots}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" scripts/srun.sh
            done
        done
    done
done