#!/bin/bash                   
#SBATCH --partition=kira-lab,overcap
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node=a40:1
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=50G
#SBATCH -x optimistprime,protocol,xaea-12,chappie,cyborg,baymax,voltron,crushinator,qt-1,shakey,cheetah,samantha,bishop,consu,heistotron,megabot,robby,chitti,kitt,megazord,nestor,omgwth

<<com
Example Slurm evaluation script. 
Notes:
- VQAv2 test-dev and test-std annotations are not publicly available. 
  To evaluate on these splits, please follow the VQAv2 instructions and submit to EvalAI.
  This script will evaluate on the val split.
com

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# export MASTER_PORT=$(shuf -i 0-65535 -n 1)
# Loop until we find an unused port
while :
do
    MASTER_PORT=$(shuf -i 1024-65535 -n 1)  # Randomly choose a port between 1024 and 65535
    # Check if the port is in use
    if ! lsof -i :$MASTER_PORT > /dev/null; then
        export MASTER_PORT
        echo "Selected available port: $MASTER_PORT"
        break
    else
        echo "Port $MASTER_PORT is already in use. Trying another..."
    fi
done

export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /coc/testnvme/chuang475/projects/VLMEvalKit

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=1 run.py \
    --model $model \
    --support_data $support_dataset \
    --query_data $query_dataset \
    --rag_method $rag_method \
    --num_shots 0 1 2 4 8 \
    --verbose \
    --reuse \
    --icl_rationale \
    # --multi_step_icl \
    # --work-dir $output_dir \
