#!/bin/bash
cd /coc/testnvme/chuang475/projects/VLMEvalKit
name="vlmeval"
model="Qwen2.5-VL-7B-Instruct"
# dataset="MathVista_MINI"
# dataset="EMMA"
# dataset="ScienceQA_TEST_QCME_wo_last"
# dataset="A-OKVQA"
# dataset="MME_CoT_TEST"
dataset="LogicVista_Rationale"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/${model}/${dataset}/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL,model=${model},dataset=${dataset},output_dir=${output_dir}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" scripts/srun.sh